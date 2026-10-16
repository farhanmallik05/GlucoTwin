"""
src/api/main.py
GlucoTwin FastAPI Application Entry Point.
Provides REST and WebSocket interfaces for clinicians, patient apps, and hospital monitors.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import logging
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("glucotwin")

from src.api.routes.health import router as health_router
from src.api.routes.patients import router as patients_router
from src.api.routes.demo import router as demo_router, seed_demo_patients
from src.api.routes.websocket import router as ws_router
from src.api.routes.fhir import router as fhir_router
from src.twin.engine import get_twin_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle hooks: seed default demo cohorts on initialization."""
    try:
        await seed_demo_patients()
        logger.info("Demo patient cohorts successfully seeded on startup.")
    except Exception as e:
        logger.warning(f"Demo seed failed during startup: {e}")
    yield


app = FastAPI(
    title="GlucoTwin — Type 2 Diabetes Digital Twin API",
    description="Real-Time Digital Twin for Adverse Glycemic Event Prediction & Clinical Explainability (CDS Tool — Not SaMD)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Cross-Origin Resource Sharing (CORS) Configuration
dashboard_origins = os.environ.get(
    "DASHBOARD_ORIGIN",
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000"
)
allow_origins_list = [orig.strip() for orig in dashboard_origins.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routes
app.include_router(health_router)
app.include_router(patients_router)
app.include_router(demo_router)
app.include_router(ws_router)
app.include_router(fhir_router)

# Prometheus-compatible Metrics Endpoint
@app.get("/metrics", tags=["Monitoring"], response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus-compatible operational and clinical twin metrics."""
    engine = get_twin_engine()
    twins = engine.twins
    total_twins = len(twins)
    calibrated_twins = sum(1 for t in twins.values() if t.is_calibrated)
    total_alerts = sum(len(t.alert_history) for t in twins.values())
    red_alerts = sum(sum(1 for a in t.alert_history if a.tier == "RED") for t in twins.values())
    orange_alerts = sum(sum(1 for a in t.alert_history if a.tier == "ORANGE") for t in twins.values())

    lines = [
        "# HELP glucotwin_active_twins Total registered patient digital twins",
        "# TYPE glucotwin_active_twins gauge",
        f"glucotwin_active_twins {total_twins}",
        "# HELP glucotwin_calibrated_twins Calibrated patient digital twins",
        "# TYPE glucotwin_calibrated_twins gauge",
        f"glucotwin_calibrated_twins {calibrated_twins}",
        "# HELP glucotwin_total_alerts Total clinical alerts dispatched",
        "# TYPE glucotwin_total_alerts counter",
        f"glucotwin_total_alerts {total_alerts}",
        "# HELP glucotwin_red_alerts Critical Tier 4 alerts dispatched",
        "# TYPE glucotwin_red_alerts counter",
        f"glucotwin_red_alerts {red_alerts}",
        "# HELP glucotwin_orange_alerts Warning Tier 3 alerts dispatched",
        "# TYPE glucotwin_orange_alerts counter",
        f"glucotwin_orange_alerts {orange_alerts}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.04")


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
