"""
EfficientNetB0 Transfer Learning — PCOS Ultrasound Classification

Reads the disjoint split index files produced by prepare_split.py:
    ml/ultrasound/splits/{train,val,test}.csv

Saves the trained model to:
    backend/models/pcos_efficientnet.h5

and the held-out test metrics to:
    ml/ultrasound/metrics.json

Usage:
    python ml/ultrasound/src/prepare_split.py   # once, to build the split
    python ml/ultrasound/src/train.py

Three things changed here relative to the original script:

1. NO manual /255 rescaling. Keras' EfficientNetB0 normalises inside the graph
   and expects raw 0-255 input. Dividing first stacked on top of that internal
   rescaling and fed the pretrained filters values ~255x smaller than the
   statistics they were trained on. backend/app/services/ultrasound_service.py
   was changed to match — the two must stay in agreement.

2. Validation and test come from images the model never trains on. The previous
   version passed data/test/ as validation_data, but data/test/ was a
   byte-identical copy of data/train/, so both the reported metrics and the
   EarlyStopping/ReduceLROnPlateau decisions were made on training data.

3. Training is two-phase: head-only with the backbone frozen, then fine-tuning
   the top of the backbone at a low learning rate. The original script froze
   the backbone and never unfroze it, despite a "Phase 1" comment implying a
   second stage; a frozen ImageNet probe underfits greyscale ultrasound.

READ THIS BEFORE QUOTING ANY NUMBER THIS SCRIPT PRINTS
------------------------------------------------------
The current data/ corpus is LEAKED: data/train/Normal and data/train/PCOS are
two different image collections that share zero image resolutions, and a single
threshold on image width classifies the held-out split at 88% accuracy. This
model reaches ~100% by reading the acquisition source, not the pathology.

Run `python ml/ultrasound/audit_leakage.py` and read
docs/03-findings.md. The metrics written by this script are only
meaningful once that audit passes on the dataset you are training against.
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPLIT_DIR = os.path.join(BASE_DIR, "ml", "ultrasound", "splits")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "backend", "models", "pcos_efficientnet.h5")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "ultrasound", "metrics.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = ["Normal", "PCOS"]  # index 0, 1
RANDOM_STATE = 42

# Phase 1 — train the head with the ImageNet backbone frozen.
HEAD_EPOCHS = 8
HEAD_LR = 1e-3

# Phase 2 — unfreeze the top of the backbone and fine-tune at a low LR.
# ImageNet features are a weak match for greyscale ultrasound texture, so the
# frozen linear probe tends to underfit. BatchNorm layers stay frozen: with a
# batch of 32 on a small dataset, updating their running statistics
# destabilises training.
FINE_TUNE = True
FINE_TUNE_EPOCHS = 12
FINE_TUNE_LR = 1e-5
FINE_TUNE_FROM_LAYER = 200  # unfreeze roughly the top third of EfficientNetB0


def build_model():
    """EfficientNetB0 with an ImageNet backbone and a small classification head."""
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=HEAD_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def unfreeze_top(model, base, from_layer: int):
    """Unfreeze the top of the backbone, keeping BatchNorm layers frozen."""
    base.trainable = True
    for layer in base.layers[:from_layer]:
        layer.trainable = False
    for layer in base.layers[from_layer:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print(f"Fine-tuning from layer {from_layer}/{len(base.layers)} "
          f"-> {trainable:,} trainable parameters")

    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_split(name: str) -> pd.DataFrame:
    path = os.path.join(SPLIT_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing {path}. Run: python ml/ultrasound/src/prepare_split.py"
        )
    df = pd.read_csv(path)
    df["filepath"] = df["filepath"].apply(lambda p: os.path.join(BASE_DIR, p))
    return df


def drop_unreadable(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Filter out images PIL cannot fully decode.

    The original script moved corrupt files into a corrupt_files/ directory,
    mutating data/ as a side effect of training. Filtering the index instead
    leaves the dataset untouched and keeps the run reproducible.
    """
    keep = []
    for path in df["filepath"]:
        try:
            with Image.open(path) as img:
                img.load()  # full decode; .convert() alone misses truncated files
            keep.append(True)
        except Exception:
            keep.append(False)
    dropped = int((~pd.Series(keep, index=df.index)).sum())
    if dropped:
        print(f"  {name}: dropped {dropped} unreadable image(s)")
    return df[pd.Series(keep, index=df.index)]


def make_generator(df, datagen, shuffle):
    # NOTE: no rescale anywhere — EfficientNetB0 normalises internally.
    return datagen.flow_from_dataframe(
        df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASSES,
        shuffle=shuffle,
        seed=RANDOM_STATE,
    )


def main():
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    print("Loading split index files...")
    train_df = drop_unreadable(load_split("train"), "train")
    val_df = drop_unreadable(load_split("val"), "val")
    test_df = drop_unreadable(load_split("test"), "test")
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}\n")

    train_aug = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_aug = ImageDataGenerator()

    train_data = make_generator(train_df, train_aug, shuffle=True)
    val_data = make_generator(val_df, eval_aug, shuffle=False)
    test_data = make_generator(test_df, eval_aug, shuffle=False)

    print(f"\nClasses: {train_data.class_indices}\n")

    model, base = build_model()

    classes = train_data.classes
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(classes), y=classes
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Class weights applied: {class_weight_dict}")

    def callbacks(patience):
        return [
            EarlyStopping(patience=patience, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-8, monitor="val_loss"),
        ]

    print("\n--- Phase 1: head only, backbone frozen ---")
    model.fit(
        train_data,
        epochs=HEAD_EPOCHS,
        validation_data=val_data,
        class_weight=class_weight_dict,
        callbacks=callbacks(4),
        verbose=1,
    )

    if FINE_TUNE:
        print("\n--- Phase 2: fine-tuning the top of the backbone ---")
        model = unfreeze_top(model, base, FINE_TUNE_FROM_LAYER)
        model.fit(
            train_data,
            epochs=FINE_TUNE_EPOCHS,
            validation_data=val_data,
            class_weight=class_weight_dict,
            callbacks=callbacks(5),
            verbose=1,
        )

    # ------ Evaluate ONCE on the held-out test split ------
    print("\n--- Held-out test evaluation ---")
    test_data.reset()
    probs = model.predict(test_data, verbose=0).flatten()
    y_true = test_data.classes
    y_pred = (probs >= 0.5).astype(int)

    loss, acc = model.evaluate(test_data, verbose=0)
    auc = float(roc_auc_score(y_true, probs))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"Test Loss:      {loss:.4f}")
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test ROC-AUC:   {auc:.4f}")
    print(f"Sensitivity:    {tp / (tp + fn):.4f}")
    print(f"Specificity:    {tn / (tn + fp):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    metrics = {
        "model": "EfficientNetB0 + GlobalAveragePooling + Dense(64) + Dense(1, sigmoid)",
        "training": {
            "phase1": {"epochs": HEAD_EPOCHS, "lr": HEAD_LR, "backbone": "frozen"},
            "phase2": (
                {"epochs": FINE_TUNE_EPOCHS, "lr": FINE_TUNE_LR,
                 "unfrozen_from_layer": FINE_TUNE_FROM_LAYER, "batchnorm": "frozen"}
                if FINE_TUNE else None
            ),
        },
        "split": "disjoint 70/15/15 by image content hash (prepare_split.py)",
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "test_loss": round(float(loss), 4),
        "test_accuracy": round(float(acc), 4),
        "test_roc_auc": round(auc, 4),
        "test_sensitivity": round(float(tp / (tp + fn)), 4),
        "test_specificity": round(float(tn / (tn + fp)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    # Attach the leakage verdict so these numbers are never read in isolation.
    leak_path = os.path.join(BASE_DIR, "ml", "ultrasound", "leakage_report.json")
    if os.path.exists(leak_path):
        with open(leak_path, encoding="utf-8") as fh:
            leak = json.load(fh)
        metrics["dataset_leakage"] = {
            "leakage_detected": leak.get("leakage_detected"),
            "conclusion": leak.get("conclusion"),
            "width_only_baseline_test_accuracy":
                leak.get("width_only_baseline", {}).get("test_accuracy"),
            "see": "docs/03-findings.md",
        }
        if leak.get("leakage_detected"):
            metrics["METRICS_ARE_NOT_VALID"] = (
                "The dataset is leaked: image resolution alone separates the classes. "
                "These scores measure acquisition source, not PCOS detection. "
                "Do not quote them."
            )
    else:
        metrics["dataset_leakage"] = "not audited - run ml/ultrasound/audit_leakage.py"

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nMetrics written to {METRICS_PATH}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
