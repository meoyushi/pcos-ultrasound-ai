import os
from PIL import Image
import numpy as np

def test_heuristic(directory, label):
    ratios = []
    for f in os.listdir(directory):
        if not f.endswith('.jpg'): continue
        path = os.path.join(directory, f)
        pil_img = Image.open(path).convert("L")
        arr = np.array(pil_img)
        h, w = arr.shape
        center = arr[h//4:3*h//4, w//4:3*w//4]
        # Ignore complete black background (pixels == 0)
        valid_pixels = center[center > 5] 
        if valid_pixels.size == 0:
            continue
        # Dark pixels are cysts (between 5 and 45)
        dark_ratio = np.sum(valid_pixels < 45) / valid_pixels.size
        ratios.append(dark_ratio)
        if len(ratios) >= 10: break
    
    print(f"--- {label} ---")
    print(f"Ratios: {ratios}")
    print(f"Mean: {np.mean(ratios):.3f}")

test_heuristic("data/test/Normal", "Normal")
test_heuristic("data/test/PCOS", "PCOS")
