import torch
import clip
import subprocess
import os
import platform
import pathlib
import argparse

def open_file_with_associated_program(filepath):
    # https://stackoverflow.com/questions/434597/open-document-with-default-os-application-in-python-both-in-windows-and-mac-os
    if platform.system() == 'Darwin':       # macOS
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':    # Windows
        os.startfile(filepath)
    else:                                   # linux variants
        subprocess.call(('xdg-open', filepath))

def find_best_match(index, text_features):
    best_sim = -100
    best_key = None

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    for key, image_features in index.items():
        sim = float(cos(image_features, text_features))
        if sim > best_sim:
            best_sim = sim
            best_key = key
    
    return best_key

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=pathlib.Path)
    args = parser.parse_args()
    image_dir = args.image_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device = {}".format(device), flush=True)

    print("loading model...", flush=True)
    model, _ = clip.load("ViT-L/14", device=device)

    print("loading index...", flush=True)
    index = torch.load(image_dir / "clip_index.pt")

    while True:
        text = input("--> ")
        text = clip.tokenize(text).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)[0]
        
        key = find_best_match(index, text_features)
        print("best_match: {}".format(key))

        open_file_with_associated_program(image_dir / key)

if __name__ == '__main__':
    main()
