"""POST /api/predict/ultrasound – Ultrasound image prediction."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.ultrasound_service import ModelUnavailableError, ultrasound_service

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/api/predict/ultrasound")
async def predict_ultrasound(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Accepted: jpg, png, webp.",
        )

    try:
        image_bytes = await file.read()
        result = ultrasound_service.predict(image_bytes)
    except ModelUnavailableError as e:
        # The model could not be loaded. Return 503 rather than a fabricated
        # prediction, so callers can tell "unavailable" from "negative".
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    return {
        "success": True,
        "mode": "ultrasound",
        **result,
    }
