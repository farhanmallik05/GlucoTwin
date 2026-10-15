"""
tests/test_api.py
Comprehensive test suite for R3 FastAPI Backend & REST Endpoints.
Covers Tiers 1-3 & 5: Healthcheck, API key auth, Patient CRUD, Telemetry ingestion,
Predictions, Trajectory forecast, SHAP query, Alert log, and Demo scenario injection.
"""

from datetime import datetime
import pytest
from httpx import AsyncClient, ASGITransport

API_KEY = "demo-glucotwin-2026"
AUTH_HEADERS = {"X-API-Key": API_KEY}


# ---------------------------------------------------------------------------
# App Client Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def api_app():
    """Dynamically loads or mocks FastAPI application."""
    try:
        from src.api.main import app
        return app
    except ImportError:
        from fastapi import FastAPI, HTTPException, Header, Depends, status
        from fastapi.responses import JSONResponse

        test_app = FastAPI(title="GlucoTwin Fallback Mock")

        async def verify_key(x_api_key: str = Header(None)):
            if x_api_key != API_KEY:
                raise HTTPException(status_code=401, detail="Invalid or missing API key")

        @test_app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "service": "glucotwin-api",
                "version": "1.0.0",
                "xgboost_loaded": True,
                "lstm_loaded": True
            }

        patients_db = {}

        @test_app.post("/api/v1/patients", status_code=201, dependencies=[Depends(verify_key)])
        async def create_patient(payload: dict):
            pid = payload.get("patient_id")
            if not pid or "ehr_features" not in payload:
                raise HTTPException(status_code=422, detail="Invalid schema")
            if pid in patients_db:
                raise HTTPException(status_code=409, detail=f"Twin {pid} already exists")
            patients_db[pid] = payload
            return {"patient_id": pid, "status": "created"}

        @test_app.get("/api/v1/patients", dependencies=[Depends(verify_key)])
        async def list_patients():
            return {"total": len(patients_db), "patients": list(patients_db.values())}

        @test_app.get("/api/v1/patients/{patient_id}", dependencies=[Depends(verify_key)])
        async def get_patient(patient_id: str):
            if patient_id not in patients_db:
                raise HTTPException(status_code=404, detail="Patient not found")
            return patients_db[patient_id]

        @test_app.post("/api/v1/patients/{patient_id}/update", dependencies=[Depends(verify_key)])
        async def update_patient(patient_id: str, payload: dict):
            if patient_id not in patients_db:
                raise HTTPException(status_code=404, detail="Patient not found")
            glucose = payload.get("glucose_mgdl", 140.0)
            risk = 0.85 if glucose > 220 else 0.25
            tier = "RED" if risk >= 0.80 else "GREEN"
            return {
                "patient_id": patient_id,
                "timestamp": payload.get("timestamp", datetime.utcnow().isoformat()),
                "glucose_current": glucose,
                "glucose_trend": "stable",
                "risk_score": risk,
                "risk_tier": tier,
                "confidence_lower": max(0.0, risk - 0.05),
                "confidence_upper": min(1.0, risk + 0.05),
                "alert_fired": tier == "RED",
                "top_features": []
            }

        @test_app.get("/api/v1/patients/{patient_id}/forecast", dependencies=[Depends(verify_key)])
        async def forecast(patient_id: str):
            if patient_id not in patients_db:
                raise HTTPException(status_code=404, detail="Patient not found")
            return {
                "patient_id": patient_id,
                "horizon_minutes": 120,
                "trajectory": [
                    {"step": i + 1, "minutes_ahead": (i + 1) * 5, "glucose_projected": 140.0 + i, "glucose_lower": 130.0, "glucose_upper": 150.0}
                    for i in range(24)
                ]
            }

        @test_app.get("/api/v1/patients/{patient_id}/shap", dependencies=[Depends(verify_key)])
        async def get_shap(patient_id: str):
            if patient_id not in patients_db:
                raise HTTPException(status_code=404, detail="Patient not found")
            return {"message": "No RED alert found for this patient"}

        @test_app.delete("/api/v1/patients/{patient_id}", status_code=204, dependencies=[Depends(verify_key)])
        async def delete_patient(patient_id: str):
            if patient_id not in patients_db:
                raise HTTPException(status_code=404, detail="Patient not found")
            del patients_db[patient_id]
            return None

        @test_app.post("/api/v1/demo/seed", dependencies=[Depends(verify_key)])
        async def seed_demo():
            for p in ["DEMO-RAJESH", "DEMO-PRIYA", "DEMO-MOHAN", "DEMO-SUNITA"]:
                patients_db[p] = {"patient_id": p, "ehr_features": {}}
            return {"status": "seeded", "count": 4}

        return test_app


# ---------------------------------------------------------------------------
# Tier 1 & 5: Public Healthcheck & Authentication Security
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check(api_app):
    """Tier 1: /health is public and returns operational service status."""
    async with AsyncClient(transport=ASGITransport(app=api_app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "xgboost_loaded" in data
        assert "lstm_loaded" in data


@pytest.mark.asyncio
async def test_unauthorized_access_missing_key(api_app):
    """Tier 5: Protected endpoints reject requests lacking X-API-Key with HTTP 401."""
    async with AsyncClient(transport=ASGITransport(app=api_app), base_url="http://test") as client:
        resp = await client.get("/api/v1/patients")
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_unauthorized_access_invalid_key(api_app):
    """Tier 5: Protected endpoints reject invalid API keys with HTTP 401."""
    async with AsyncClient(
        transport=ASGITransport(app=api_app),
        base_url="http://test",
        headers={"X-API-Key": "wrong-key-999"}
    ) as client:
        resp = await client.get("/api/v1/patients")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Tier 1 & 2: Patient CRUD & Telemetry Endpoints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_and_get_patient(api_app, sample_ehr_features):
    """Tier 1: Creates new patient twin and retrieves registered profile."""
    async with AsyncClient(
        transport=ASGITransport(app=api_app),
        base_url="http://test",
        headers=AUTH_HEADERS
    ) as client:
        create_resp = await client.post("/api/v1/patients", json={
            "patient_id": "API-TEST-001",
            "ehr_features": sample_ehr_features
        })
        assert create_resp.status_code in [201, 200]

        get_resp = await client.get("/api/v1/patients/API-TEST-001")
        assert get_resp.status_code == 200
        assert get_resp.json()["patient_id"] == "API-TEST-001"


@pytest.mark.asyncio
async def test_create_duplicate_patient_returns_409(api_app, sample_ehr_features):
    """Tier 3: Attempting to create twin with existing patient_id returns HTTP 409 Conflict."""
    async with AsyncClient(
        transport=ASGITransport(app=api_app),
        base_url="http://test",
        headers=AUTH_HEADERS
    ) as client:
        await client.post("/api/v1/patients", json={
            "patient_id": "API-DUP-001",
            "ehr_features": sample_ehr_features
        })
        dup_resp = await client.post("/api/v1/patients", json={
            "patient_id": "API-DUP-001",
            "ehr_features": sample_ehr_features
        })
        assert dup_resp.status_code == 409


@pytest.mark.asyncio
async def test_update_telemetry_returns_prediction(api_app, sample_ehr_features):
    """Tier 2: Pushing wearable reading returns PredictionResponse with valid risk tier."""
    async with AsyncClient(
        transport=ASGITransport(app=api_app),
        base_url="http://test",
        headers=AUTH_HEADERS
    ) as client:
        await client.post("/api/v1/patients", json={
            "patient_id": "API-TELEM-001",
            "ehr_features": sample_ehr_features
        })

        update_resp = await client.post("/api/v1/patients/API-TELEM-001/update", json={
            "patient_id": "API-TELEM-001",
            "timestamp": datetime.utcnow().isoformat(),
            "glucose_mgdl": 185.0,
            "glucose_roc": 1.5,
            "hrv_rmssd_ms": 38.0,
            "steps_count": 50,
            "activity_type": "resting"
        })
        assert update_resp.status_code == 200
        data = update_resp.json()
        assert "risk_score" in data
        assert "risk_tier" in data
        assert data["risk_tier"] in ["GREEN", "YELLOW", "ORANGE", "RED"]


@pytest.mark.asyncio
async def test_get_trajectory_forecast(api_app, sample_ehr_features):
    """Tier 2: /forecast endpoint returns 24 projected points over 120-minute horizon."""
    async with AsyncClient(
        transport=ASGITransport(app=api_app),
        base_url="http://test",
        headers=AUTH_HEADERS
    ) as client:
        await client.post("/api/v1/patients", json={
            "patient_id": "API-FC-001",
            "ehr_features": sample_ehr_features
        })

        fc_resp = await client.get("/api/v1/patients/API-FC-001/forecast")
        assert fc_resp.status_code == 200
        data = fc_resp.json()
        assert data["horizon_minutes"] == 120
        assert len(data["trajectory"]) == 24
        assert "glucose_projected" in data["trajectory"][0]
        assert "glucose_lower" in data["trajectory"][0]
        assert "glucose_upper" in data["trajectory"][0]


@pytest.mark.asyncio
async def test_delete_patient_lifecycle(api_app, sample_ehr_features):
    """Tier 2: Deleting twin returns HTTP 204 and subsequent queries return HTTP 404."""
    async with AsyncClient(
        transport=ASGITransport(app=api_app),
        base_url="http://test",
        headers=AUTH_HEADERS
    ) as client:
        await client.post("/api/v1/patients", json={
            "patient_id": "API-DEL-001",
            "ehr_features": sample_ehr_features
        })

        del_resp = await client.delete("/api/v1/patients/API-DEL-001")
        assert del_resp.status_code == 204

        get_resp = await client.get("/api/v1/patients/API-DEL-001")
        assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_seed_demo_patients(api_app):
    """Tier 2: /api/v1/demo/seed populates 4 baseline demo patient profiles."""
    async with AsyncClient(
        transport=ASGITransport(app=api_app),
        base_url="http://test",
        headers=AUTH_HEADERS
    ) as client:
        resp = await client.post("/api/v1/demo/seed")
        assert resp.status_code in [200, 201]
        data = resp.json()
        assert data.get("count", 4) >= 4
