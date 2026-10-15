"""
src/api/main.py
GlucoTwin FastAPI Application Entry Point.
Provides REST and WebSocket interfaces for clinicians, patient apps, and hospital monitors.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from src.api.routes.health import router as health_router
from src.api.routes.patients import router as patients_router
from src.api.routes.demo import router as demo_router, seed_demo_patients
from src.api.routes.websocket import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle hooks: seed default demo cohorts on initialization."""
    try:
        await seed_demo_patients()
    except Exception:
        pass
    yield


app = FastAPI(
    title="GlucoTwin — Type 2 Diabetes Digital Twin API",
    description="Real-Time Digital Twin for Adverse Glycemic Event Prediction & Clinical Explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Cross-Origin Resource Sharing (CORS) Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routes
app.include_router(health_router)
app.include_router(patients_router)
app.include_router(demo_router)
app.include_router(ws_router)

# Serve Frontend Dashboard if directory exists
dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "public")
if os.path.exists(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")


@app.get("/", tags=["Root"])
async def root():
    """Root redirect and gateway information."""
    return {
        "project": "GlucoTwin — Type 2 Diabetes Digital Twin",
        "challenge": "Happiest Health Digital Twin Challenge 2026",
        "status": "online",
        "api_docs": "/docs",
        "health_check": "/health",
        "dashboard_ui": "/dashboard/"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
