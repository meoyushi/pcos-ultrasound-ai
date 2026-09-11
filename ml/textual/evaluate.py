"""
Cross-validate the textual (clinical questionnaire) Random Forest.

Imports the SAME feature list, preprocessing and hyperparameters the API
serves, so the numbers here describe the deployed model rather than a
different one trained on columns the form never collects.

Writes ml/textual/metrics.json.

Usage:
    python ml/textual/evaluate.py
"""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from app.services.textual_service import (  # noqa: E402
    FORM_FEATURES,
    MODEL_PARAMS,
    load_dataset,
)

METRICS_PATH = os.path.join(BASE_DIR, "ml", "textual", "metrics.json")

N_SPLITS = 5
N_REPEATS = 5
CV_SEED = 7

SCORING = ["accuracy", "roc_auc", "recall", "precision", "f1"]


def main():
    X, y = load_dataset()
    print(f"Dataset: {len(X)} patients, {X.shape[1]} features")
    print(f"Class balance: {int((y == 0).sum())} negative / {int((y == 1).sum())} positive")
    print(f"Majority-class baseline accuracy: {max((y == 0).mean(), (y == 1).mean()):.4f}\n")
    print(f"Features ({len(FORM_FEATURES)}): {', '.join(FORM_FEATURES)}\n")

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=CV_SEED
    )
    model = RandomForestClassifier(**MODEL_PARAMS)
    results = cross_validate(model, X, y, cv=cv, scoring=SCORING)

    print(f"{N_SPLITS}-fold stratified CV, {N_REPEATS} repeats "
          f"({N_SPLITS * N_REPEATS} fits):\n")
    summary = {}
    for metric in SCORING:
        scores = results[f"test_{metric}"]
        summary[metric] = {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
        }
        print(f"  {metric:10s} {scores.mean():.4f} +/- {scores.std():.4f}")

    # Feature importances from a fit on the full dataset (descriptive only)
    model.fit(X, y)
    importances = sorted(
        zip(X.columns, model.feature_importances_), key=lambda kv: -kv[1]
    )
    print("\nFeature importances:")
    for name, value in importances:
        print(f"  {name:24s} {value:.4f}")

    metrics = {
        "model": "RandomForestClassifier",
        "hyperparameters": {k: v for k, v in MODEL_PARAMS.items()},
        "features": list(X.columns),
        "n_features": int(X.shape[1]),
        "n_samples": int(len(X)),
        "class_balance": {
            "negative": int((y == 0).sum()),
            "positive": int((y == 1).sum()),
        },
        "majority_class_baseline_accuracy": round(
            float(max((y == 0).mean(), (y == 1).mean())), 4
        ),
        "cv": {
            "type": "RepeatedStratifiedKFold",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "random_state": CV_SEED,
        },
        "scores": summary,
        "feature_importances": {name: round(float(v), 4) for name, v in importances},
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nMetrics written to {os.path.relpath(METRICS_PATH, BASE_DIR)}")


if __name__ == "__main__":
    main()
