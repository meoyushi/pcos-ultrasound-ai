"""
Build a genuine train/val/test split for the ultrasound images.

Why this exists
---------------
`data/train/` and `data/test/` are NOT a split: every file in data/test/ is a
byte-identical copy of a file in data/train/ (verified by MD5 over all 1922
test files). Training with `validation_data=test_data` therefore validated on
the training images, so no reported accuracy from that setup was a
generalisation estimate.

This script rebuilds an honest split:

  1. Walks both directories and hashes every image (MD5 of file contents).
  2. Collapses duplicates — 3846 files reduce to 1921 unique images.
  3. Verifies that no image content appears under more than one class label.
  4. Writes a stratified, deterministic 70/15/15 split as CSV index files.

It does NOT copy or move any image, so `data/` is left untouched and the
140 MB of pixels are not duplicated on disk. `train.py` consumes the CSVs via
`flow_from_dataframe`.

Usage:
    python ml/ultrasound/src/prepare_split.py
"""

import hashlib
import os
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SOURCE_DIRS = [os.path.join(BASE_DIR, "data", "train"), os.path.join(BASE_DIR, "data", "test")]
SPLIT_DIR = os.path.join(BASE_DIR, "ml", "ultrasound", "splits")

CLASSES = ["Normal", "PCOS"]
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_STATE = 42

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def collect_unique_images() -> pd.DataFrame:
    """Hash every image and keep one representative path per unique content."""
    by_hash = defaultdict(list)

    for source in SOURCE_DIRS:
        for cls in CLASSES:
            directory = os.path.join(source, cls)
            if not os.path.isdir(directory):
                continue
            for name in sorted(os.listdir(directory)):
                path = os.path.join(directory, name)
                if not os.path.isfile(path) or not name.lower().endswith(IMAGE_EXTS):
                    continue
                with open(path, "rb") as fh:
                    digest = hashlib.md5(fh.read()).hexdigest()
                by_hash[digest].append((cls, path))

    total_files = sum(len(v) for v in by_hash.values())

    # A single image appearing under two different labels would poison the split.
    conflicts = [d for d, entries in by_hash.items() if len({c for c, _ in entries}) > 1]
    if conflicts:
        raise SystemExit(
            f"{len(conflicts)} image(s) appear under both Normal and PCOS. "
            "Resolve these label conflicts before splitting."
        )

    records = []
    for digest, entries in by_hash.items():
        cls, path = entries[0]  # sorted walk => deterministic representative
        records.append(
            {
                "md5": digest,
                "filepath": os.path.relpath(path, BASE_DIR).replace("\\", "/"),
                "label": cls,
                "duplicate_count": len(entries),
            }
        )

    df = pd.DataFrame(records).sort_values("md5").reset_index(drop=True)
    print(f"Scanned {total_files} files across {len(SOURCE_DIRS)} directories.")
    print(f"Collapsed to {len(df)} unique images ({total_files - len(df)} duplicates removed).")
    return df


def main():
    df = collect_unique_images()

    # Stratified 70 / 15 / 15. Split off test first, then val from the remainder.
    train_val, test = train_test_split(
        df, test_size=TEST_FRACTION, stratify=df["label"], random_state=RANDOM_STATE
    )
    val_ratio = VAL_FRACTION / (1.0 - TEST_FRACTION)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val["label"], random_state=RANDOM_STATE
    )

    os.makedirs(SPLIT_DIR, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out = os.path.join(SPLIT_DIR, f"{name}.csv")
        part.sort_values("md5").to_csv(out, index=False)
        counts = part["label"].value_counts().to_dict()
        print(f"  {name:5s} n={len(part):5d}  {counts}  -> {os.path.relpath(out, BASE_DIR)}")

    # Prove the split is disjoint rather than asserting it in a comment.
    sets = {n: set(p["md5"]) for n, p in [("train", train), ("val", val), ("test", test)]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sets[a] & sets[b]
        assert not overlap, f"{a}/{b} overlap: {len(overlap)} images"
    print("\nVerified: train, val and test are pairwise disjoint by image content.")


if __name__ == "__main__":
    main()
