"""POST /api/predict/textual – Clinical features prediction."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.textual_service import textual_service

router = APIRouter()


class TextualRequest(BaseModel):
    features: dict


@router.post("/api/predict/textual")
async def predict_textual(body: TextualRequest):
    try:
        result = textual_service.predict(body.features)
    except ValueError as e:
        # Deliberate, user-facing validation messages from TextualService
        # (missing field, non-numeric value, non-positive height/weight).
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed.")

    return {
        "success": True,
        "mode": "textual",
        **result,
    }
