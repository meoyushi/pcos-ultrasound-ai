"""
Ultrasound image prediction service.

Serves an EfficientNetB0 model transfer-learned on PCOS ultrasound images.

IMPORTANT — input scaling
-------------------------
Keras' EfficientNetB0 performs its own input normalisation *inside the graph*
(a Rescaling + Normalization stack immediately after the input layer) and
therefore expects raw 0-255 pixel values. This service used to divide by 255
before calling the model, which stacked on top of the internal rescaling and
fed the pretrained filters values ~255x smaller than the statistics they were
trained on. That is now fixed here and in ml/ultrasound/src/train.py.

Both files must stay in agreement: whatever `_preprocess` does here has to
match the training generator exactly, or predictions silently degrade.

If the .h5 model file is missing or fails to load, the service enters MOCK
mode and `predict()` raises ModelUnavailableError instead of returning a
fabricated result. `is_degraded()` exposes the same state to health checks.
"""

import io
import os

import numpy as np
from PIL import Image

# Robust path detection for Render/Local
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODEL_PATH = os.path.join(_BASE_DIR, "models", "pcos_efficientnet.h5")

IMG_SIZE = (224, 224)

# ---------------------------------------------------------------------------
# Legacy pixel-variance fallback — DISABLED BY DEFAULT.
#
# An earlier revision overrode the network whenever its output fell inside
# (0.1, 0.9), substituting a hand-tuned function of the image's grayscale
# standard deviation. That made the deployed predictor a one-feature pixel
# statistic rather than EfficientNetB0, so it is off by default.
#
# The thresholds below were hand-fitted on data/test/, which at the time was a
# byte-identical copy of data/train/ — so its apparent accuracy was measured on
# the images it was tuned on. Do not re-enable without evaluating it on a
# genuinely held-out split.
# ---------------------------------------------------------------------------
ENABLE_PIXEL_STD_FALLBACK = False
_FALLBACK_BAND = (0.1, 0.9)
_FALLBACK_INTERCEPT = 35.0
_FALLBACK_SLOPE = 25.0


class ModelUnavailableError(RuntimeError):
    """Raised when the ultrasound model cannot serve a real prediction.

    Previously this condition returned a hard-coded 'No PCOS / 50%' response
    with HTTP 200, which was indistinguishable from a genuine result. Callers
    should surface this as 503 rather than fabricate a prediction.
    """


class UltrasoundService:
    """Singleton – lazy-loads the Keras model on first predict() call."""

    def __init__(self):
        self.model = None
        self._model_loaded = False
        self._mock_mode = False
        self._load_error = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        if self._model_loaded:
            return

        if not os.path.exists(_MODEL_PATH):
            print(
                f"[UltrasoundService] WARNING: Model file not found at {_MODEL_PATH}. "
                "Running in MOCK mode — ultrasound predictions are NOT real."
            )
            self._load_error = "model file not found"
            self._mock_mode = True
            self._model_loaded = True
            return

        try:
            from tensorflow.keras.models import load_model  # type: ignore

            self.model = load_model(_MODEL_PATH)
            self._model_loaded = True
            print("[UltrasoundService] Model loaded successfully.")
        except Exception as e:
            err_msg = str(e)
            print(
                f"[UltrasoundService] Failed to load model: {err_msg}. "
                "Running in MOCK mode — ultrasound predictions are NOT real."
            )
            self._load_error = err_msg
            self._mock_mode = True
            self._model_loaded = True

    def is_degraded(self) -> bool:
        """True when the service cannot serve real predictions (mock mode).

        Loads the model on first call so that health checks report the true
        state rather than 'not yet attempted'.
        """
        self._load_model()
        return self._mock_mode

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Read image bytes and resize to 224x224.

        Pixels are left in the 0-255 range on purpose — EfficientNetB0
        normalises internally. See the module docstring.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        return np.expand_dims(arr, axis=0)  # batch dim

    def _pixel_std_probability(self, image_bytes: bytes) -> float:
        """Legacy fallback: map grayscale std into a pseudo-probability."""
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
        arr = np.array(pil_img)

        # Exclude near-black pixels (ultrasound border/background)
        valid_pixels = arr[arr > 5]
        pixel_std = 0.0 if valid_pixels.size == 0 else float(np.std(valid_pixels))

        prob = (pixel_std - _FALLBACK_INTERCEPT) / _FALLBACK_SLOPE
        return float(np.clip(prob, 0.02, 0.98))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, image_bytes: bytes) -> dict:
        """
        Predict PCOS from an ultrasound image.
        Returns { prediction, label, confidence, pcos_probability, mock }.
        """
        self._load_model()

        if self._mock_mode:
            raise ModelUnavailableError(
                f"Ultrasound model unavailable ({self._load_error}). "
                "Refusing to return a fabricated prediction."
            )

        img_tensor = self._preprocess(image_bytes)
        prob = float(self.model.predict(img_tensor, verbose=0)[0][0])
        used_fallback = False

        if ENABLE_PIXEL_STD_FALLBACK and _FALLBACK_BAND[0] < prob < _FALLBACK_BAND[1]:
            print(f"[UltrasoundService] Weak prediction ({prob:.3f}). Using pixel-std fallback.")
            prob = self._pixel_std_probability(image_bytes)
            used_fallback = True

        prediction = 1 if prob >= 0.5 else 0
        confidence = max(prob, 1 - prob) * 100

        return {
            "prediction": prediction,
            "label": "PCOS Detected" if prediction == 1 else "No PCOS",
            "confidence": round(confidence, 1),
            "pcos_probability": round(prob * 100, 1),
            "mock": False,
            "used_pixel_std_fallback": used_fallback,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
ultrasound_service = UltrasoundService()
