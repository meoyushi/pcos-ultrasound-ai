from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import textual, ultrasound, combined
from app.services.ultrasound_service import ultrasound_service

app = FastAPI(
    title="PCOS Multimodal Prediction API",
    description="Predict PCOS using textual clinical data, ultrasound images, or a combined approach.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS – allow the Vite dev server and any deployed frontend origin
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(textual.router)
app.include_router(ultrasound.router)
app.include_router(combined.router)


@app.get("/")
async def health_check():
    """Health check that reflects real serving capability.

    Reports "degraded" when the ultrasound model failed to load, so a deploy
    that silently lost its .h5 is visible instead of reporting healthy while
    every ultrasound request 503s.
    """
    ultrasound_ok = not ultrasound_service.is_degraded()
    return {
        "status": "healthy" if ultrasound_ok else "degraded",
        "service": "PCOS Multimodal Prediction API",
        "version": "1.0.0",
        "models": {
            "textual": "ready",
            "ultrasound": "ready" if ultrasound_ok else "unavailable",
        },
    }
