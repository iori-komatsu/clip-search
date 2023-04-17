from flask import Flask, request
import chevron
import torch
import clip
import subprocess
import os
import platform
import pathlib
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
index = None

def init_app():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=pathlib.Path)
    args = parser.parse_args()

    image_dir = args.image_dir

    print("loading index...", flush=True)
    global index
    index = torch.load(image_dir / "clip_index.pt")

    print("loading model...", flush=True)
    global model
    model, _ = clip.load("ViT-L/14", device=device)

    return Flask(__name__, static_folder=image_dir, static_url_path='/static')

app = init_app()

def open_file_with_associated_program(filepath):
    # https://stackoverflow.com/questions/434597/open-document-with-default-os-application-in-python-both-in-windows-and-mac-os
    if platform.system() == 'Darwin':       # macOS
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':    # Windows
        os.startfile(filepath)
    else:                                   # linux variants
        subprocess.call(('xdg-open', filepath))

def find_best_match(index, text_features):
    ranking = []

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    for key, image_features in index.items():
        sim = float(cos(image_features, text_features))
        ranking.append((sim, key))
    
    ranking.sort(reverse=True)
    return ranking


TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
    <title>Local CLIP search</title>
</head>
<body>

    <div class="container" style="margin-top: 2em">
        <div class="mb-3 row">
            <div class="col">
                <form method="POST" action="/search">
                    <div class="input-group">
                        <input class="form-control" type="text" name="q" value="{{query}}" placeholder="Query" autofocus>
                        <input class="btn btn-primary" type="submit">
                    </div>
                </form>
            </div>
        </div>

        <div class="m-3 row">
            <div class="col">
                <ol>
                {{#images}}
                    <li>
                        <a href="/static/{{key}}" target="_blank"><img src="/static/{{key}}" width="450"></a>
                        sim = {{sim}}
                    </li>
                {{/images}}
                </ol>
            </div>
        </div>
    </div>

</body>
</html>
"""

@app.route("/")
def root():
    return chevron.render(TEMPLATE, {
        "images": [],
        "query": "",
    })

@app.route("/search", methods=["POST"])
def api_search():
    query = request.form["q"]
    ranking = search(query)
    ranking = ranking[:100]
    return chevron.render(TEMPLATE, {
        "images": [
            {"key": key, "sim": sim}
            for sim, key in ranking
        ],
        "query": query,
    })

def search(query):
    text = clip.tokenize(query).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)[0]
    
    return find_best_match(index, text_features)

app.run(host="127.0.0.1", port=3000)
