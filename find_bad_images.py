import os
from PIL import Image

def find_bad(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.startswith('.'):
                continue
            path = os.path.join(root, f)
            try:
                with Image.open(path) as img:
                    img.load()
            except Exception as e:
                print(f"Bad file: {path} - {e}")
                
find_bad("data/train")
find_bad("data/test")
