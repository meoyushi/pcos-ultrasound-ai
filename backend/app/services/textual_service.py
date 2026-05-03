"""
Textual (clinical features) prediction service.

Ported from the original Flask app's PCOSPredictor class.
Trains a Random Forest classifier on startup from the PCOS dataset,
then exposes a predict() method that accepts a partial feature dict
(user-facing fields only) and fills missing features with dataset medians.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "PCOS_data_without_infertility.xlsx"
)


class TextualService:
    """Singleton-style service – instantiate once at module level."""

    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.feature_names: list[str] = []
        self.median_defaults: dict[str, float] = {}
        self._train()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train(self):
        df = pd.read_excel(_DATA_PATH, sheet_name="Full_new")

        # Drop non-feature columns
        drop_cols = ["Sl. No", "Patient File No.", "Unnamed: 44"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # Strip leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()

        # ------ Calculate BMI if missing or null ------
        if "BMI" not in df.columns or df["BMI"].isnull().any():
            df["BMI"] = df["Weight (Kg)"] / ((df["Height(Cm)"] / 100) ** 2)

        # ------ Impute missing values ------
        # Convert object columns like 'AMH(ng/mL)' to numeric first
        for col in df.columns:
            if df[col].dtype == 'object' and col != "PCOS (Y/N)":
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for col in df.columns:
            if col == "PCOS (Y/N)":
                continue
            imputer = SimpleImputer(strategy="median")
            df[col] = imputer.fit_transform(df[[col]]).ravel()

        # ------ Separate features / target ------
        X = df.drop("PCOS (Y/N)", axis=1)
        y = df["PCOS (Y/N)"]

        self.feature_names = [c.strip() for c in X.columns.tolist()]
        self.median_defaults = X.median().to_dict()

        # ------ Train Random Forest ------
        # Use more estimators and max_depth for better stability
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        self.model.fit(X, y)

        print(f"[TextualService] Trained on {len(X)} samples, {len(self.feature_names)} features.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features_dict: dict) -> dict:
        """
        Accept a partial features dict (e.g. 14 user-facing fields).
        Fill missing features with dataset medians, then predict.
        """
        # Prepare the input row
        input_data = {}
        
        # 1. Start with medians
        for fname, fval in self.median_defaults.items():
            input_data[fname.strip()] = float(fval)

        # 2. Overlay user data (case-insensitive and whitespace-stripped matching)
        for user_key, user_val in features_dict.items():
            stripped_user_key = user_key.strip().lower()
            for model_key in self.feature_names:
                if model_key.lower() == stripped_user_key:
                    input_data[model_key] = float(user_val)
                    break
        
        # 3. Handle BMI specifically (Weight / Height^2)
        weight = input_data.get("Weight (Kg)")
        height = input_data.get("Height(Cm)")
        if weight and height:
            input_data["BMI"] = weight / ((height / 100) ** 2)

        # Create the vector in the exact order the model expects
        row_dict = {fname: [input_data[fname]] for fname in self.feature_names}
        df_row = pd.DataFrame(row_dict)

        prediction = int(self.model.predict(df_row)[0])
        probabilities = self.model.predict_proba(df_row)[0]

        pcos_prob = float(probabilities[1]) * 100
        # Confidence is the probability of the predicted class
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
