import argparse
import pathlib
import torch
import clip
import glob
from PIL import Image

def list_images(image_dir):
    return glob.glob("**/*.jpg", root_dir=image_dir, recursive=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    image_dir = args.image_dir
    batch_size = args.batch_size

    image_files = list_images(image_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device = {}".format(device), flush=True)

    print("loading model...", flush=True)
    model, preprocess = clip.load("ViT-L/14", device=device)

    n = len(image_files)
    result = {}
    for start in range(0, n, batch_size):
        progress = "[{}/{}]".format(start+1, n)
        targets = image_files[start:start+batch_size]

        print("{} preprocess...".format(progress), flush=True)
        images = []
        for image_file in targets:
            with Image.open(image_dir / image_file) as f:
                images.append(preprocess(f).unsqueeze(0).to(device))
        images = torch.cat(images, dim=0)

        print("{} calculate prob...".format(progress))
        with torch.no_grad():
            image_features = model.encode_image(images)
        
        for i in range(len(targets)):
            result[targets[i]] = image_features[i]

    torch.save(result, image_dir / "clip_index.pt")
    print("Done")

if __name__ == "__main__":
    main()
