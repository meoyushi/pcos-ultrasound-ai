"""
EfficientNetB0 Transfer Learning — PCOS Ultrasound Classification

Trains on the ultrasound dataset at:
    data/train/{Normal, PCOS}/
    data/test/{Normal, PCOS}/

Saves the trained model to:
    backend/models/pcos_efficientnet.h5

Usage:
    python ml/ultrasound/src/train.py
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from PIL import Image

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "backend", "models", "pcos_efficientnet.h5")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3


def build_model():
    """Build EfficientNetB0 with custom classification head."""
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))

    # Phase 1: Completely freeze the base model
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE), 
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def main():
    print(f"Training dir: {TRAIN_DIR}")
    print(f"Test dir:     {TEST_DIR}")
    print(f"Model path:   {MODEL_SAVE_PATH}")

    # ------ Verify and Skip Corrupt Images ------
    def check_images(directory):
        bad_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, file)
                    try:
                        with Image.open(path) as img:
                            img.convert('RGB')
                    except Exception:
                        bad_files.append(path)
        return bad_files

    print("Checking for corrupt images (this may take a moment)...")
    bad_train = check_images(TRAIN_DIR)
    bad_test = check_images(TEST_DIR)
    
    if bad_train or bad_test:
        print(f"Found {len(bad_train) + len(bad_test)} corrupt images. Moving them to exclude from training.")
        for f in bad_train + bad_test:
            try:
                corrupt_dir = os.path.join(os.path.dirname(f), "corrupt_files")
                os.makedirs(corrupt_dir, exist_ok=True)
                dest = os.path.join(corrupt_dir, os.path.basename(f))
                os.rename(f, dest)
                print(f"  - Moved corrupt file: {os.path.basename(f)}")
            except Exception as e:
                print(f"  - Failed to move {f}: {e}")

    # ------ Data generators with augmentation ------
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=["Normal", "PCOS"],
    )

    test_data = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=["Normal", "PCOS"],
        shuffle=False,
    )

    print(f"\nTraining samples: {train_data.samples}")
    print(f"Test samples:     {test_data.samples}")
    print(f"Classes:          {train_data.class_indices}\n")

    # ------ Build & train ------
    model, base = build_model()
    model.summary()

    # Calculate class weights to handle imbalance
    classes = train_data.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights applied: {class_weight_dict}")

    print("\n--- Training (Base Frozen with Class Weights) ---")
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-7, monitor="val_loss")
    ]

    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=test_data,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # ------ Evaluate ------
    loss, acc = model.evaluate(test_data, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    # Classification report
    test_data.reset()
    preds = (model.predict(test_data, verbose=0) >= 0.5).astype(int).flatten()
    true_labels = test_data.classes[: len(preds)]
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=["Normal", "PCOS"]))

    # ------ Save model ------
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
