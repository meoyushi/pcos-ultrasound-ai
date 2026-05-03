import os
from PIL import Image
import numpy as np

def test_heuristic(directory, label):
    means = []
    stds = []
    for f in os.listdir(directory):
        if not f.endswith('.jpg'): continue
        path = os.path.join(directory, f)
        pil_img = Image.open(path).convert("L")
        arr = np.array(pil_img)
        # ignore pure black
        valid = arr[arr > 5]
        if valid.size == 0: continue
        means.append(np.mean(valid))
        stds.append(np.std(valid))
        if len(means) >= 10: break
    
    print(f"--- {label} ---")
    print(f"Mean of Means: {np.mean(means):.3f}")
    print(f"Mean of Stds: {np.mean(stds):.3f}")

test_heuristic("data/test/Normal", "Normal")
test_heuristic("data/test/PCOS", "PCOS")
