# Vibe Assist 08 — API Backend
## FastAPI REST + WebSocket Implementation

> **Vibe Coding Session 08/10** | Estimated time: 4 hours
> Goal: Production-quality FastAPI backend with REST endpoints + real-time WebSocket.

---

## 🎯 Session Goal

By the end of this session you will have:
- ✅ FastAPI app with all REST endpoints working
- ✅ WebSocket endpoint broadcasting twin updates every 5 seconds (demo mode)
- ✅ API key authentication middleware
- ✅ CORS configured for React dashboard
- ✅ Auto-generated OpenAPI docs at `/docs`
- ✅ Demo patient seeded and accessible via API
- ✅ All routes tested with httpx

---

## API Design Overview

```
Base URL: http://localhost:8000

Authentication: X-API-Key header (all endpoints except /health)

Endpoints:
  GET  /health                           → System health
  GET  /docs                             → OpenAPI UI (auto-generated)

  POST /api/v1/patients                  → Create patient twin
  GET  /api/v1/patients                  → List all twins
  GET  /api/v1/patients/{id}             → Get twin state + summary
  POST /api/v1/patients/{id}/update      → Push wearable reading
  GET  /api/v1/patients/{id}/predict     → Get latest prediction
  GET  /api/v1/patients/{id}/history     → Glucose + prediction history (24h)
  GET  /api/v1/patients/{id}/forecast    → 2-hour glucose trajectory
  GET  /api/v1/patients/{id}/shap        → SHAP explanation (last RED alert)
  GET  /api/v1/patients/{id}/alerts      → Alert history
  DELETE /api/v1/patients/{id}           → Remove twin

  WS   /ws/patients/{id}                → Real-time twin updates
  
  POST /api/v1/demo/inject-event         → Demo: inject scenario event
  POST /api/v1/demo/start-scenario       → Demo: start auto-scenario
  POST /api/v1/demo/seed                 → Demo: seed demo patients
```

---

## 🤖 Vibe Prompt 1 — FastAPI App Entry Point

```
Create src/api/main.py — the FastAPI application entry point.

Requirements:
1. FastAPI app with title "GlucoTwin API", version "1.0.0"
2. Lifespan context manager that:
   - On startup: loads ML models from artifacts/, creates TwinEngine,
     seeds demo patients, starts wearable stream manager
   - On shutdown: saves all twin states to SQLite
3. CORS middleware: allow localhost:3000 (React dashboard)
4. Middleware to log all requests with timing
5. Global exception handler returning JSON errors
6. Include routers from: routes/patients.py, routes/websocket.py, 
   routes/health.py, routes/demo.py
7. Add rate limiting: 100 requests/minute per API key

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger("glucotwin.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("🚀 GlucoTwin API starting up...")
    
    # 1. Load ML models
    from src.models.xgboost_model import XGBoostAEPredictor
    from src.models.lstm_model import LSTMAEPredictor
    from src.models.ensemble import EnsemblePredictor
    from src.explainability.shap_explainer import SHAPExplainer
    
    app.state.xgb = XGBoostAEPredictor.load("src/models/artifacts/xgboost_v1.pkl")
    app.state.lstm = LSTMAEPredictor.load("src/models/artifacts/lstm_v1.pt")
    app.state.ensemble = EnsemblePredictor(app.state.xgb, app.state.lstm)
    app.state.shap = SHAPExplainer(app.state.xgb)
    
    # 2. Initialize TwinEngine
    from src.twin.engine import TwinEngine
    app.state.engine = TwinEngine(app.state.xgb, app.state.lstm, app.state.shap)
    
    # 3. Initialize persistence
    from src.twin.persistence import TwinPersistence
    app.state.db = TwinPersistence()
    
    # 4. Seed demo patients
    from src.api.demo import seed_demo_patients
    await seed_demo_patients(app.state.engine, app.state.db)
    
    # 5. Start wearable stream manager
    from src.data.ingestion.wearable_stream import WearableStreamManager
    app.state.stream_manager = WearableStreamManager(demo_mode=True, 
                                                       speed_multiplier=60.0)
    
    logger.info("✅ GlucoTwin API ready — Demo patients seeded")
    
    yield  # Application runs here
    
    # Shutdown: save state
    logger.info("💾 Saving twin states...")
    for twin in app.state.engine.twins.values():
        app.state.db.save_twin_state(twin)
    logger.info("👋 GlucoTwin API shut down cleanly")

app = FastAPI(
    title="GlucoTwin API",
    description="Digital Twin API for Type 2 Diabetes Adverse Event Prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from src.api.routes import patients, websocket, health, demo
app.include_router(patients.router, prefix="/api/v1", tags=["Patients"])
app.include_router(websocket.router, tags=["WebSocket"])
app.include_router(health.router, tags=["Health"])
app.include_router(demo.router, prefix="/api/v1/demo", tags=["Demo"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

Also create src/api/config.py with Settings (using pydantic-settings):
    API_KEY: str
    DEMO_MODE: bool = True
    TWIN_UPDATE_INTERVAL_SECONDS: int = 5
    DATABASE_URL: str = "sqlite:///./glucotwin.db"
    LOG_LEVEL: str = "INFO"
    MODEL_ARTIFACTS_PATH: str = "src/models/artifacts"
```

---

## 🤖 Vibe Prompt 2 — Patient Routes

```
Create src/api/routes/patients.py with all patient CRUD + prediction endpoints.

All routes require API key auth (via dependency injection).

from fastapi import APIRouter, Depends, HTTPException
from src.api.middleware.auth import verify_api_key

router = APIRouter()

@router.post("/patients", status_code=201)
async def create_patient_twin(
    patient: PatientCreateRequest,
    engine: TwinEngine = Depends(get_engine),
    db: TwinPersistence = Depends(get_db),
    _: None = Depends(verify_api_key),
) → PatientSummaryResponse:
    """Create a new patient digital twin."""
    if engine.get_twin(patient.patient_id):
        raise HTTPException(409, f"Twin {patient.patient_id} already exists")
    twin = engine.create_twin(patient.patient_id, patient.ehr_features)
    return PatientSummaryResponse(**twin.get_summary())

@router.get("/patients")
async def list_patient_twins(engine: TwinEngine = Depends(get_engine),
                               _: None = Depends(verify_api_key)):
    """List all active patient twins with risk summaries."""
    summaries = engine.get_all_twin_summaries()
    return {"total": len(summaries), "patients": summaries}

@router.get("/patients/{patient_id}")
async def get_patient_twin(patient_id: str, ...):
    """Get full twin state for a patient."""

@router.post("/patients/{patient_id}/update")
async def update_patient_twin(patient_id: str, 
                               reading: WearableReadingRequest, ...):
    """Push a new wearable reading to the patient twin."""
    result = await engine.process_reading(reading.to_wearable_reading())
    await db.log_prediction(result)
    # Broadcast to WebSocket clients
    await ws_manager.broadcast(patient_id, result)
    return PredictionResponse.from_result(result)

@router.get("/patients/{patient_id}/predict")
async def get_prediction(patient_id: str, ...):
    """Get latest prediction without pushing new data."""

@router.get("/patients/{patient_id}/history")
async def get_history(patient_id: str, hours: int = 24, ...):
    """Get glucose + prediction history as time series."""
    history = await db.get_prediction_history(patient_id, hours)
    return {"patient_id": patient_id, "hours": hours, "data": history}

@router.get("/patients/{patient_id}/forecast")
async def get_forecast(patient_id: str, ...):
    """Get 2-hour glucose trajectory forecast."""
    twin = get_twin_or_404(engine, patient_id)
    return twin.get_glucose_forecast()

@router.get("/patients/{patient_id}/shap")
async def get_shap_explanation(patient_id: str, ...):
    """Get SHAP explanation for the most recent RED alert."""
    twin = get_twin_or_404(engine, patient_id)
    if not twin.last_alert or twin.last_alert.tier != "RED":
        return {"message": "No RED alert found for this patient"}
    return twin.last_alert.shap_features

@router.get("/patients/{patient_id}/alerts")
async def get_alerts(patient_id: str, ...):
    """Get alert history for a patient."""
    return await db.get_alert_history(patient_id)

@router.delete("/patients/{patient_id}", status_code=204)
async def delete_patient_twin(patient_id: str, ...):
    """Remove patient twin."""

Create Pydantic schemas in src/api/schemas/:
  patient.py: PatientCreateRequest, PatientSummaryResponse
  prediction.py: WearableReadingRequest, PredictionResponse
  
All schemas use Pydantic v2 with model_validator and field_validator.
Include example values in Field() for OpenAPI documentation.
```

---

## 🤖 Vibe Prompt 3 — WebSocket Endpoint

```
Create src/api/routes/websocket.py with WebSocket manager.

The WebSocket provides real-time twin updates to the React dashboard.
Each patient has their own channel: ws://localhost:8000/ws/patients/{id}

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio, json

router = APIRouter()

class WebSocketManager:
    """Manages WebSocket connections per patient."""
    
    def __init__(self):
        # patient_id → set of connected WebSocket clients
        self.connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, patient_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections.setdefault(patient_id, set()).add(websocket)
    
    def disconnect(self, patient_id: str, websocket: WebSocket):
        if patient_id in self.connections:
            self.connections[patient_id].discard(websocket)
    
    async def broadcast(self, patient_id: str, result: PredictionResult):
        """Broadcast prediction result to all connected clients for a patient."""
        if patient_id not in self.connections:
            return
        message = {
            "type": "twin_update",
            "patient_id": patient_id,
            "timestamp": result.timestamp.isoformat(),
            "glucose_current": result.glucose_current,
            "glucose_trend": result.glucose_trend,
            "risk_score": round(result.risk_score, 3),
            "risk_tier": result.risk_tier,
            "alert_fired": result.alert_fired,
            "alert_message": result.alert.message if result.alert else None,
            "top_features": result.top_features or [],
            "confidence_lower": result.confidence_lower,
            "confidence_upper": result.confidence_upper,
        }
        dead_connections = set()
        for ws in self.connections[patient_id].copy():
            try:
                await ws.send_json(message)
            except Exception:
                dead_connections.add(ws)
        for ws in dead_connections:
            self.disconnect(patient_id, ws)
    
    async def send_personal_message(self, patient_id: str, 
                                     websocket: WebSocket, 
                                     message: dict):
        """Send message to a single WebSocket client."""
        await websocket.send_json(message)

ws_manager = WebSocketManager()

@router.websocket("/ws/patients/{patient_id}")
async def websocket_endpoint(patient_id: str, websocket: WebSocket,
                              engine: TwinEngine = Depends(get_engine)):
    twin = engine.get_twin(patient_id)
    if not twin:
        await websocket.close(code=4004, reason="Patient not found")
        return
    
    await ws_manager.connect(patient_id, websocket)
    
    # Send current state immediately on connection
    await ws_manager.send_personal_message(patient_id, websocket, {
        "type": "connected",
        "patient_id": patient_id,
        "current_state": twin.get_summary(),
    })
    
    try:
        while True:
            # Keep connection alive — wait for client ping or disconnect
            data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            if data == "ping":
                await websocket.send_text("pong")
    except asyncio.TimeoutError:
        await websocket.send_text("pong")  # Keepalive
    except WebSocketDisconnect:
        ws_manager.disconnect(patient_id, websocket)

Also, create a background task that drives the wearable stream 
and auto-updates twins in demo mode:

async def demo_stream_loop(engine: TwinEngine, 
                            stream_manager: WearableStreamManager,
                            ws_manager: WebSocketManager,
                            db: TwinPersistence):
    """Background task: stream wearable data and update twins."""
    async for reading in stream_manager.stream_all_patients():
        result = await engine.process_reading(reading)
        await db.log_prediction(result)
        if result.alert_fired:
            await db.log_alert(result.alert)
        await ws_manager.broadcast(reading.patient_id, result)
        await asyncio.sleep(0)  # Yield to event loop
```

---

## 🤖 Vibe Prompt 4 — Demo Routes

```
Create src/api/routes/demo.py with demo-specific endpoints 
for the hackathon demonstration.

@router.post("/seed")
async def seed_demo_patients(engine, db):
    """
    Create 4 demo patients with different risk profiles:
    1. DEMO-RAJESH: High risk, post-lunch hyperglycaemia scenario
    2. DEMO-PRIYA: Low risk, well-controlled patient
    3. DEMO-MOHAN: Medium risk, insulin-dependent, labile glucose
    4. DEMO-SUNITA: Elderly, brittle, frequent hypoglycaemia risk
    """

@router.post("/inject-event")
async def inject_event(patient_id: str, event_type: str, engine, ws_manager):
    """
    Inject a specific event for a demo patient.
    event_type: "post_meal_spike" | "insulin_overdose" | "exercise_burst"
    
    This is for live demos — presenter can trigger dramatic events.
    """
    stream_manager = get_stream_manager()
    await stream_manager.inject_event(patient_id, event_type)
    return {"message": f"Injected {event_type} for {patient_id}"}

@router.post("/start-scenario")
async def start_scenario(patient_id: str = "DEMO-RAJESH"):
    """Start the post-lunch hyperglycaemia scenario (auto-plays over 2 min)."""

@router.post("/reset")
async def reset_demo(engine):
    """Reset all demo patients to baseline state."""

@router.get("/events")
async def list_injectable_events():
    """List all available injectable demo events."""
    return {
        "events": [
            {"id": "post_meal_spike", "description": "Simulates eating a large meal without correction", "duration_minutes": 90},
            {"id": "insulin_overdose", "description": "Simulates insulin dose error → hypoglycaemia", "duration_minutes": 45},
            {"id": "exercise_burst", "description": "Simulates vigorous exercise → glucose drop", "duration_minutes": 30},
        ]
    }
```

---

## 🤖 Vibe Prompt 5 — API Tests

```
Create tests/test_api.py with comprehensive API tests using httpx AsyncClient.

import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app

API_KEY = "test-key-123"

@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), 
        base_url="http://test",
        headers={"X-API-Key": API_KEY}
    ) as ac:
        yield ac

async def test_health_check(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "xgboost_loaded" in data
    assert "lstm_loaded" in data

async def test_create_patient(client):
    resp = await client.post("/api/v1/patients", json={
        "patient_id": "TEST-001",
        "ehr_features": sample_ehr_features(),
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["patient_id"] == "TEST-001"

async def test_create_duplicate_patient(client):
    await client.post("/api/v1/patients", json={"patient_id": "TEST-DUP", ...})
    resp = await client.post("/api/v1/patients", json={"patient_id": "TEST-DUP", ...})
    assert resp.status_code == 409

async def test_update_patient_returns_prediction(client):
    await client.post("/api/v1/patients", json={"patient_id": "TEST-002", ...})
    resp = await client.post("/api/v1/patients/TEST-002/update", json={
        "patient_id": "TEST-002",
        "timestamp": "2026-10-01T10:00:00",
        "glucose_mgdl": 185.3,
        "glucose_roc": 2.1,
        # ...
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "risk_score" in data
    assert "risk_tier" in data
    assert data["risk_tier"] in ["GREEN", "YELLOW", "ORANGE", "RED"]

async def test_unauthorized_access(client):
    no_auth_client = AsyncClient(transport=ASGITransport(app=app), 
                                  base_url="http://test")
    resp = await no_auth_client.get("/api/v1/patients")
    assert resp.status_code == 401

async def test_get_history(client):
    # Push 10 readings first
    for i in range(10):
        await client.post("/api/v1/patients/DEMO-RAJESH/update", json=...)
    resp = await client.get("/api/v1/patients/DEMO-RAJESH/history?hours=1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) > 0

async def test_websocket_connection():
    from starlette.testclient import TestClient
    client = TestClient(app, headers={"X-API-Key": API_KEY})
    with client.websocket_connect("/ws/patients/DEMO-RAJESH") as ws:
        data = ws.receive_json()
        assert data["type"] == "connected"
        assert data["patient_id"] == "DEMO-RAJESH"

Run with: pytest tests/test_api.py -v --asyncio-mode=auto
```

---

## ✅ Manual API Verification

```bash
# Start the API
cd GlucoTwin
uvicorn src.api.main:app --reload --port 8000

# In another terminal:
export API_KEY="your-key-from-.env"

# Health check
curl http://localhost:8000/health

# List demo patients
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/patients

# Get specific patient
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/patients/DEMO-RAJESH

# Inject demo event
curl -X POST -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/api/v1/demo/inject-event?patient_id=DEMO-RAJESH&event_type=post_meal_spike"

# OpenAPI docs
open http://localhost:8000/docs
```

---

## 📋 Session 08 Completion Checklist

- [ ] `main.py` starts up cleanly, models loaded
- [ ] Demo patients seeded on startup
- [ ] All REST endpoints return correct status codes
- [ ] WebSocket connects and receives initial state
- [ ] Demo inject-event endpoint triggers visible state change
- [ ] API key auth works (401 without key)
- [ ] OpenAPI docs accessible at /docs
- [ ] CORS allows localhost:3000
- [ ] All API tests pass
- [ ] Background stream loop auto-updates twins every 5 seconds

## 🔜 Next Session

→ **09-dashboard-ui.md** — React dashboard with real-time glucose charts

---

*GlucoTwin Vibe Assist 08 | API Backend | Digital Twin Challenge 2026*
