"""POST /api/predict/ultrasound – Ultrasound image prediction."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.ultrasound_service import ultrasound_service

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
        return {
            "success": True,
            "mode": "ultrasound",
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
