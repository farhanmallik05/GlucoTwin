"""
tests/test_upgraded_features.py
Validation test suite for 10/10 Remediation Upgrades:
- Alert fatigue suppression persistence across restarts
- Sensor artifact cascade mitigation & plausibility clamping
- Dynamic SHAP explainability & MC Dropout credible intervals
- HL7 FHIR Patient Bundle ingestion
- Prometheus operational metrics endpoint
- Security hardening (no key leakage in 401 detail, environment override)
"""

from datetime import datetime, timedelta, timezone
import pytest
from httpx import AsyncClient, ASGITransport

from src.twin.patient_twin import PatientTwin, get_utc_now
from src.twin.alert_system import AlertSystem, Alert
from src.twin.persistence import TwinPersistence
from src.data.ingestion.wearable_simulator import WearableReading
from src.models.ensemble import EnsemblePredictor
from src.models.xgboost_model import XGBoostAEPredictor
from src.models.lstm_model import LSTMAEPredictor
from src.explainability.shap_explainer import SHAPExplainer
from src.api.main import app

AUTH_HEADERS = {"X-API-Key": "demo-glucotwin-2026"}


# ---------------------------------------------------------------------------
# 1. Alert Persistence & Fatigue Suppression Recovery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_alert_persistence_and_suppression_recovery(sample_ehr_features):
    """Verifies that alert history persisted to SQLite survives restart and preserves fatigue cooldown."""
    db = TwinPersistence(database_url="sqlite:///:memory:")
    await db.initialize()

    now = get_utc_now()
    t0 = now - timedelta(minutes=10)

    # 1. Generate an initial RED alert
    alert1 = Alert(
        patient_id="PERSIST-FATIGUE-01",
        timestamp=t0,
        tier="RED",
        risk_score=0.88,
        event_type="hyperglycaemia",
        predicted_event_in_minutes=45,
        message="Critical excursion predicted"
    )

    # 2. Persist to DB table alert_logs
    await db.log_alert(alert1)

    # 3. Simulate process restart: query alert_logs from DB
    persisted_records = await db.get_recent_alerts(patient_id="PERSIST-FATIGUE-01")
    assert len(persisted_records) == 1
    assert persisted_records[0]["tier"] == "RED"
    assert persisted_records[0]["patient_id"] == "PERSIST-FATIGUE-01"

    # 4. Initialize fresh AlertSystem and load persisted records
    new_alert_system = AlertSystem(fatigue_suppression_minutes=30)
    new_alert_system.load_history_from_records(persisted_records)

    # 5. Process duplicate RED alert 10 minutes later (within 30m window) -> must be suppressed!
    t1 = t0 + timedelta(minutes=10)
    suppressed = new_alert_system.process_alert(
        patient_id="PERSIST-FATIGUE-01",
        timestamp=t1,
        risk_score=0.90,
        glucose=260.0,
        roc=2.5
    )
    assert suppressed is None, "Fatigue suppression must suppress duplicate alert within 30 min after restart"

    # 6. Process RED alert 35 minutes later -> should fire!
    t2 = t0 + timedelta(minutes=35)
    fired = new_alert_system.process_alert(
        patient_id="PERSIST-FATIGUE-01",
        timestamp=t2,
        risk_score=0.92,
        glucose=275.0,
        roc=3.0
    )
    assert fired is not None
    assert fired.tier == "RED"
    await db.close()


# ---------------------------------------------------------------------------
# 2. Sensor Artifact Cascade Mitigation & Plausibility Clamping
# ---------------------------------------------------------------------------

def test_sensor_artifact_damping_and_data_quality_suppression(sample_ehr_features):
    """Verifies that non-physiological glucose jumps (>20 mg/dL in 5m) are clamped on first reading."""
    twin = PatientTwin(patient_id="SENSOR-TEST-01", ehr_features=sample_ehr_features)
    now = get_utc_now()

    # Feed 15 baseline readings at 120 mg/dL (establishes steady state past cold start)
    for i in range(15):
        twin.update(WearableReading(
            patient_id="SENSOR-TEST-01",
            timestamp=now + timedelta(minutes=i * 5),
            glucose_mgdl=120.0,
            glucose_roc=0.0
        ))

    # Single-step artifact jump: sensor spikes from 120 to 240 mg/dL (jump of 120 mg/dL)
    jump_reading = WearableReading(
        patient_id="SENSOR-TEST-01",
        timestamp=now + timedelta(minutes=80),
        glucose_mgdl=240.0,
        glucose_roc=15.0
    )
    result = twin.update(jump_reading)

    # First unconfirmed outlier should be clamped to current + 20 = 140 mg/dL (not 240!)
    assert twin.current_glucose == 140.0
    assert twin._unconfirmed_artifact is True
    # Clamped glucose of 140 mg/dL should NOT trigger an immediate RED emergency alert!
    assert result.risk_tier != "RED"

    # Test data_quality suppression flag
    noisy_reading = WearableReading(
        patient_id="SENSOR-TEST-01",
        timestamp=now + timedelta(minutes=85),
        glucose_mgdl=280.0,
        glucose_roc=0.0,
        data_quality="sensor_noise_compression"
    )
    noisy_res = twin.update(noisy_reading)
    assert noisy_res.alert_fired is False
    assert noisy_res.risk_tier == "GREEN"


# ---------------------------------------------------------------------------
# 3. Dynamic SHAP Explainability & MC Dropout Uncertainty
# ---------------------------------------------------------------------------

def test_dynamic_shap_attribution_and_mc_dropout(sample_ehr_features):
    """Verifies that SHAP features are dynamically derived and confidence intervals are bounded."""
    twin = PatientTwin(patient_id="SHAP-TEST-01", ehr_features=sample_ehr_features)
    xgb = XGBoostAEPredictor()
    lstm = LSTMAEPredictor()
    ens = EnsemblePredictor(xgb_model=xgb, lstm_model=lstm)
    shap_exp = SHAPExplainer(model=xgb)

    twin._xgb_model = xgb
    twin._lstm_model = lstm
    twin._ensemble_model = ens
    twin._shap_explainer = shap_exp

    # Feed readings to reach critical acute spike
    now = get_utc_now()
    for i in range(20):
        g = 140.0 + (i * 7.0)  # Ascends past 260 mg/dL
        res = twin.update(WearableReading(
            patient_id="SHAP-TEST-01",
            timestamp=now + timedelta(minutes=i * 5),
            glucose_mgdl=g,
            glucose_roc=2.5,
            hrv_rmssd_ms=20.0
        ))

    assert res.risk_tier == "RED"
    assert res.top_features is not None
    assert len(res.top_features) > 0

    # Ensure dynamic attribution contains required explainability fields
    top_feat = res.top_features[0]
    assert "feature" in top_feat
    assert "clinical_label" in top_feat
    assert "shap_value" in top_feat
    assert "direction" in top_feat
    assert isinstance(top_feat["shap_value"], float)

    # Ensure Bayesian credible intervals are dynamically bounded
    assert 0.0 <= res.confidence_lower <= res.risk_score
    assert res.risk_score <= res.confidence_upper <= 1.0


# ---------------------------------------------------------------------------
# 4. HL7 FHIR Bundle Ingestion Endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fhir_bundle_ingestion_api():
    """Verifies POST /api/v1/fhir/patient-bundle ingests Patient and Observation resources."""
    bundle_payload = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "IN-DEL-9842",
                    "name": [{"given": ["Aarav"], "family": "Patel"}],
                    "gender": "male",
                    "birthDate": "1972-04-15"
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4"}]},
                    "valueQuantity": {"value": 8.6, "unit": "%"}
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "2339-0"}]},
                    "valueQuantity": {"value": 182.0, "unit": "mg/dL"}
                }
            }
        ]
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS) as client:
        resp = await client.post("/api/v1/fhir/patient-bundle", json=bundle_payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "success"
        assert data["patient_id"] == "FHIR-IN-DEL-9842"
        assert data["demographics"]["name"] == "Aarav Patel"
        assert data["demographics"]["hba1c"] == 8.6


# ---------------------------------------------------------------------------
# 5. Prometheus Operational Metrics Endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prometheus_metrics_endpoint():
    """Verifies GET /metrics outputs compliant Prometheus plaintext format."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "glucotwin_active_twins" in text
        assert "glucotwin_calibrated_twins" in text
        assert "glucotwin_total_alerts" in text
        assert "glucotwin_red_alerts" in text


# ---------------------------------------------------------------------------
# 6. Security Hardening: Redacted 401 Detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_security_401_no_key_leak():
    """Verifies 401 Unauthorized response does not disclose secret API keys."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/patients", headers={"X-API-Key": "invalid-hacker-key"})
        assert resp.status_code == 401
        detail = resp.json().get("detail", "")
        # Must not contain the valid key string!
        assert "demo-glucotwin-2026" not in detail
