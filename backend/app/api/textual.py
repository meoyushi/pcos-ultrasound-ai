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
        return {
            "success": True,
            "mode": "textual",
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
