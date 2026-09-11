"""POST /api/predict/combined – Weighted ensemble of textual + ultrasound."""

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.textual_service import textual_service
from app.services.ultrasound_service import ModelUnavailableError, ultrasound_service

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Ensemble weights
TEXTUAL_WEIGHT = 0.6
ULTRASOUND_WEIGHT = 0.4


@router.post("/api/predict/combined")
async def predict_combined(
    features: str = Form(...),
    file: UploadFile = File(...),
):
    # --- Parse features JSON string ---
    try:
        features_dict = json.loads(features)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in 'features' field.")

    # --- Validate the image up front, before doing any inference work ---
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Accepted: jpg, png, webp.",
        )

    # --- Textual prediction ---
    try:
        textual_result = textual_service.predict(features_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Textual prediction failed: {e}")

    # --- Ultrasound prediction ---
    try:
        image_bytes = await file.read()
        ultrasound_result = ultrasound_service.predict(image_bytes)
    except ModelUnavailableError as e:
        # Never blend a fabricated ultrasound probability into the ensemble —
        # the result would look multimodal while being 0.6*textual + 20.
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    # --- Weighted ensemble ---
    combined_pcos_prob = (
        textual_result["pcos_probability"] * TEXTUAL_WEIGHT
        + ultrasound_result["pcos_probability"] * ULTRASOUND_WEIGHT
    )
    combined_prediction = 1 if combined_pcos_prob >= 50 else 0
    combined_confidence = abs(combined_pcos_prob - 50) * 2  # scale 0-100

    return {
        "success": True,
        "mode": "combined",
        "prediction": combined_prediction,
        "label": "PCOS Detected" if combined_prediction == 1 else "No PCOS",
        "confidence": round(combined_confidence, 1),
        "pcos_probability": round(combined_pcos_prob, 1),
        "breakdown": {
            "textual": {
                "mode": "textual",
                **textual_result,
            },
            "ultrasound": {
                "mode": "ultrasound",
                **ultrasound_result,
            },
        },
    }
