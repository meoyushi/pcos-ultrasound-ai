"""
Textual (clinical features) prediction service.

Trains a Random Forest classifier on startup from the PCOS dataset, then
exposes a predict() method for the clinical questionnaire.

Training/serving parity
-----------------------
The model is trained on exactly the features the form collects — the 13 fields
in frontend/src/utils/fields.js, plus BMI derived from weight and height.

It previously trained on all 41 columns of the spreadsheet and then, at
inference, filled the 28 the form does not collect with dataset medians. Those
28 included `Follicle No. (L)` and `Follicle No. (R)` — antral follicle counts
read off an ultrasound, and the two strongest features in the data by a wide
margin. Pinning them to a constant meant the forest's dominant split always
routed toward "Normal": measured over 5-fold CV, that setup scored 0.747
accuracy at 0.259 sensitivity, i.e. it missed roughly three of every four PCOS
cases. Training on the served feature set instead scores 0.840 accuracy at
0.731 sensitivity.

Anything added to FORM_FEATURES must also be collected by the form, or the
same train/serve mismatch reappears.

Run `python ml/textual/evaluate.py` to reproduce the cross-validation metrics.
"""

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA_PATH = os.path.join(_BASE_DIR, "data", "PCOS_data_without_infertility.xlsx")

TARGET = "PCOS (Y/N)"

# The 13 questionnaire fields (see frontend/src/utils/fields.js) plus derived BMI.
FORM_FEATURES = [
    # Demographics
    "Age (yrs)",
    "Weight (Kg)",
    "Height(Cm)",
    "BMI",
    "Blood Group",
    # Menstrual
    "Cycle(R/I)",
    "Cycle length(days)",
    # Symptoms
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    # Lifestyle
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)",
]

# Depth-limited: unbounded trees memorised all 541 rows (training accuracy 1.000)
# and scored 0.820 in CV; capping depth at 6 gives 0.840 and materially better
# sensitivity (0.646 -> 0.731).
MODEL_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    random_state=42,
    class_weight="balanced",
)


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load and clean the spreadsheet, returning (X, y) over FORM_FEATURES.

    Shared with ml/textual/evaluate.py so that the evaluation script measures
    exactly what the service serves.
    """
    df = pd.read_excel(_DATA_PATH, sheet_name="Full_new")

    # Drop identifiers and the empty trailing column
    drop_cols = ["Sl. No", "Patient File No.", "Unnamed: 44"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Header names carry stray whitespace (' Age (yrs)', 'Height(Cm) ')
    df.columns = df.columns.str.strip()

    # BMI is null in 299 of 541 rows — recompute it for every row from the
    # weight/height the form collects, so it is consistent with inference.
    df["BMI"] = df["Weight (Kg)"] / ((df["Height(Cm)"] / 100) ** 2)

    # Some numeric columns arrive as objects (e.g. a stray non-numeric cell)
    for col in df.columns:
        if df[col].dtype == "object" and col != TARGET:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    missing = [c for c in FORM_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected feature column(s): {missing}")

    X = df[FORM_FEATURES].copy()
    for col in X.columns:
        imputer = SimpleImputer(strategy="median")
        X[col] = imputer.fit_transform(X[[col]]).ravel()

    return X, df[TARGET]


class TextualService:
    """Singleton-style service – instantiate once at module level."""

    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.feature_names: list[str] = list(FORM_FEATURES)
        self._train()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train(self):
        X, y = load_dataset()
        self.feature_names = list(X.columns)
        self.model = RandomForestClassifier(**MODEL_PARAMS)
        self.model.fit(X, y)
        print(
            f"[TextualService] Trained on {len(X)} samples, "
            f"{len(self.feature_names)} features (form-only feature set)."
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features_dict: dict) -> dict:
        """Predict from the questionnaire fields.

        Every model feature must be supplied by the caller; BMI is derived from
        weight and height. A missing field is an error rather than a silent
        median substitution — that substitution is what previously collapsed
        sensitivity.
        """
        # Case-insensitive, whitespace-tolerant matching of caller keys
        supplied = {str(k).strip().lower(): v for k, v in features_dict.items()}

        row = {}
        missing = []
        for name in self.feature_names:
            if name == "BMI":
                continue  # derived below
            key = name.strip().lower()
            if key not in supplied or supplied[key] in (None, ""):
                missing.append(name)
                continue
            try:
                row[name] = float(supplied[key])
            except (TypeError, ValueError):
                raise ValueError(f"Field '{name}' must be numeric, got {supplied[key]!r}.")

        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}")

        height_cm = row["Height(Cm)"]
        weight_kg = row["Weight (Kg)"]
        if height_cm <= 0:
            raise ValueError("Height must be greater than zero.")
        if weight_kg <= 0:
            raise ValueError("Weight must be greater than zero.")
        row["BMI"] = weight_kg / ((height_cm / 100) ** 2)

        df_row = pd.DataFrame([[row[name] for name in self.feature_names]],
                              columns=self.feature_names)

        prediction = int(self.model.predict(df_row)[0])
        probabilities = self.model.predict_proba(df_row)[0]

        pcos_prob = float(probabilities[1]) * 100
        confidence = float(probabilities[prediction]) * 100

        return {
            "prediction": prediction,
            "label": "PCOS Detected" if prediction == 1 else "No PCOS",
            "confidence": round(confidence, 1),
            "pcos_probability": round(pcos_prob, 1),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
textual_service = TextualService()
