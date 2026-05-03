from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import textual, ultrasound, combined

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
    return {
        "status": "healthy",
        "service": "PCOS Multimodal Prediction API",
        "version": "1.0.0",
    }
