"""
Leakage audit for the ultrasound dataset.

WHY THIS EXISTS
---------------
After fixing the preprocessing bug and building a genuinely disjoint split, an
EfficientNetB0 with a FROZEN backbone and an 82k-parameter head reached
val_accuracy = 1.0000 after a single epoch. That is not a good result — a
frozen ImageNet probe cannot learn ovarian pathology in one epoch. It means the
two class directories are separable by something other than pathology.

This script tests that hypothesis directly. It checks whether trivial file
metadata — resolution, format, file size — predicts the label, and fits a
single-threshold rule on image width alone as a sanity baseline.

Run it before believing ANY accuracy number from this dataset:

    python ml/ultrasound/audit_leakage.py

Writes ml/ultrasound/leakage_report.json.
"""

import json
import os

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPLIT_DIR = os.path.join(BASE_DIR, "ml", "ultrasound", "splits")
REPORT_PATH = os.path.join(BASE_DIR, "ml", "ultrasound", "leakage_report.json")

POSITIVE_LABEL = "PCOS"


def load_metadata() -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        path = os.path.join(SPLIT_DIR, f"{split}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Run prepare_split.py first.")
        df = pd.read_csv(path)
        for filepath, label in zip(df["filepath"], df["label"]):
            abspath = os.path.join(BASE_DIR, filepath)
            try:
                with Image.open(abspath) as im:
                    width, height = im.size
                    fmt, mode = im.format, im.mode
            except Exception:
                continue
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "width": width,
                    "height": height,
                    "resolution": f"{width}x{height}",
                    "format": fmt,
                    "mode": mode,
                    "bytes": os.path.getsize(abspath),
                }
            )
    return pd.DataFrame(rows)


def main():
    df = load_metadata()
    print(f"Audited {len(df)} images\n")

    normal_res = set(df[df["label"] != POSITIVE_LABEL]["resolution"])
    pcos_res = set(df[df["label"] == POSITIVE_LABEL]["resolution"])
    shared = normal_res & pcos_res

    print("--- Resolution overlap between classes ---")
    print(f"  Normal-only resolutions : {len(normal_res - shared)}")
    print(f"  PCOS-only resolutions   : {len(pcos_res - shared)}")
    print(f"  Shared resolutions      : {len(shared)}")
    shared_images = int(df["resolution"].isin(shared).sum())
    print(f"  Images at a shared resolution: {shared_images}/{len(df)} "
          f"({100 * shared_images / len(df):.1f}%)\n")

    for label, group in df.groupby("label"):
        print(f"  {label:7s} n={len(group):5d}  "
              f"widths {group['width'].min()}-{group['width'].max()}  "
              f"heights {group['height'].min()}-{group['height'].max()}  "
              f"median size {int(group['bytes'].median()):,} bytes")
    print()

    # Single-threshold rule on width alone, fitted on train, scored on test.
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y_train = (train["label"] == POSITIVE_LABEL).astype(int)
    y_test = (test["label"] == POSITIVE_LABEL).astype(int)

    best_threshold, best_acc = None, -1.0
    for threshold in sorted(train["width"].unique()):
        acc = accuracy_score(y_train, (train["width"] <= threshold).astype(int))
        if acc > best_acc:
            best_threshold, best_acc = int(threshold), float(acc)

    test_acc = float(accuracy_score(y_test, (test["width"] <= best_threshold).astype(int)))
    baseline = float(max((y_test == 0).mean(), (y_test == 1).mean()))

    print("--- Trivial baseline: threshold on IMAGE WIDTH only ---")
    print(f"  Rule fitted on train: predict {POSITIVE_LABEL} if width <= {best_threshold}")
    print(f"  train accuracy            : {best_acc:.4f}")
    print(f"  HELD-OUT TEST accuracy    : {test_acc:.4f}")
    print(f"  majority-class baseline   : {baseline:.4f}")

    leaked = len(shared) == 0 or test_acc > baseline + 0.15

    print()
    if leaked:
        print("  VERDICT: LEAKED. File metadata alone separates the classes, so the")
        print("  two directories are different image collections, not one cohort.")
        print("  No accuracy figure from this dataset measures PCOS detection.")
    else:
        print("  VERDICT: no obvious metadata leakage detected.")

    report = {
        "n_images": int(len(df)),
        "resolution_overlap": {
            "normal_only": len(normal_res - shared),
            "pcos_only": len(pcos_res - shared),
            "shared": len(shared),
            "images_at_shared_resolution": shared_images,
        },
        "per_class": {
            label: {
                "n": int(len(g)),
                "width_range": [int(g["width"].min()), int(g["width"].max())],
                "height_range": [int(g["height"].min()), int(g["height"].max())],
                "median_bytes": int(g["bytes"].median()),
                "formats": g["format"].value_counts().to_dict(),
            }
            for label, g in df.groupby("label")
        },
        "width_only_baseline": {
            "rule": f"predict {POSITIVE_LABEL} if width <= {best_threshold}",
            "train_accuracy": round(best_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "majority_class_baseline": round(baseline, 4),
        },
        "leakage_detected": bool(leaked),
        "conclusion": (
            "Image resolution alone separates the two classes with zero overlap. "
            "Any model trained on this data can score near-perfectly by learning "
            "the acquisition source rather than the pathology. No PCOS detection "
            "performance claim is supportable from this dataset."
            if leaked else
            "No metadata-based separation found."
        ),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport written to {os.path.relpath(REPORT_PATH, BASE_DIR)}")


if __name__ == "__main__":
    main()
