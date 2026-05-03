"""POST /api/predict/combined – Weighted ensemble of textual + ultrasound."""

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.textual_service import textual_service
from app.services.ultrasound_service import ultrasound_service

router = APIRouter()

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

    # --- Textual prediction ---
    try:
        textual_result = textual_service.predict(features_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Textual prediction failed: {e}")

    # --- Ultrasound prediction ---
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Accepted: jpg, png, webp.",
        )

    try:
        image_bytes = await file.read()
        ultrasound_result = ultrasound_service.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ultrasound prediction failed: {e}")

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
