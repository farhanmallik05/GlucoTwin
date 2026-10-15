"""
src/api/routes/patients.py
Patient Twin CRUD, Telemetry Streaming, Trajectory Forecasting, and Explainability Endpoints.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from src.api.middleware.auth import verify_api_key
from src.api.schemas.patient import (
    ForecastResponse,
    ForecastStep,
    PatientCreateRequest,
    PatientResponse,
    PredictionResponse,
    TelemetryUpdateRequest,
)
from src.twin.engine import get_twin_engine
from src.twin.forecaster import GlucoseForecaster
from src.data.ingestion.wearable_simulator import WearableReading

router = APIRouter(prefix="/api/v1/patients", tags=["Patients"])
forecaster = GlucoseForecaster()


@router.post("", status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_api_key)])
async def create_patient(payload: Dict[str, Any]):
    """Register a new Patient Twin with baseline EHR demographics."""
    pid = payload.get("patient_id")
    ehr = payload.get("ehr_features")
    if not pid or ehr is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Fields 'patient_id' and 'ehr_features' are required."
        )

    engine = get_twin_engine()
    if engine.get_twin(pid) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Patient Twin '{pid}' already registered."
        )

    engine.create_twin(pid, ehr)
    return {"patient_id": pid, "status": "created"}


@router.get("", dependencies=[Depends(verify_api_key)])
async def list_patients():
    """Retrieve all active registered patient twins."""
    engine = get_twin_engine()
    summaries = engine.get_all_twin_summaries()
    return {"total": len(summaries), "patients": summaries}


@router.get("/{patient_id}", dependencies=[Depends(verify_api_key)])
async def get_patient(patient_id: str):
    """Fetch current state summary and physiological calibration for a specific twin."""
    engine = get_twin_engine()
    twin = engine.get_twin(patient_id)
    if twin is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient Twin '{patient_id}' not found."
        )
    return {
        "patient_id": twin.patient_id,
        "ehr_features": twin.ehr_features,
        "summary": twin.get_summary()
    }


@router.post("/{patient_id}/update", dependencies=[Depends(verify_api_key)])
async def update_patient_telemetry(patient_id: str, payload: Dict[str, Any]):
    """Ingest new wearable sensor telemetry and compute real-time prediction result."""
    engine = get_twin_engine()
    twin = engine.get_twin(patient_id)
    if twin is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient Twin '{patient_id}' not found."
        )

    glucose = float(payload.get("glucose_mgdl", 120.0))
    roc = float(payload.get("glucose_roc", 0.0))
    hrv = float(payload.get("hrv_rmssd_ms", 38.0))
    ts_str = payload.get("timestamp")
    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.now(timezone.utc).replace(tzinfo=None)

    reading = WearableReading(
        patient_id=patient_id,
        timestamp=ts,
        glucose_mgdl=glucose,
        glucose_roc=roc,
        glucose_acc=float(payload.get("glucose_acc", 0.0)),
        heart_rate_bpm=float(payload.get("heart_rate_bpm", 72.0)),
        hrv_rmssd_ms=hrv,
        hrv_sdnn_ms=float(payload.get("hrv_sdnn_ms", 45.0)),
        hrv_lf_hf_ratio=float(payload.get("hrv_lf_hf_ratio", 1.2)),
        steps_last_5min=int(payload.get("steps_last_5min", payload.get("steps_count", 0))),
        activity_type=str(payload.get("activity_type", "resting")),
    )

    result = twin.update(reading)

    return {
        "patient_id": result.patient_id,
        "timestamp": result.timestamp.isoformat() if hasattr(result.timestamp, "isoformat") else str(result.timestamp),
        "glucose_current": result.glucose_current,
        "glucose_trend": result.glucose_trend,
        "risk_score": result.risk_score,
        "risk_tier": result.risk_tier,
        "confidence_lower": result.confidence_lower,
        "confidence_upper": result.confidence_upper,
        "alert_fired": result.alert_fired,
        "top_features": result.top_features or []
    }


@router.get("/{patient_id}/forecast", dependencies=[Depends(verify_api_key)])
async def get_trajectory_forecast(patient_id: str, horizon_minutes: int = 120):
    """Retrieve 24-step forward projection (120 minutes) of glucose excursion trajectory."""
    engine = get_twin_engine()
    twin = engine.get_twin(patient_id)
    if twin is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient Twin '{patient_id}' not found."
        )

    current_g = twin.current_glucose if twin.current_glucose > 0 else 140.0
    current_roc = twin.roc_history[-1] if len(twin.roc_history) > 0 else 0.0

    steps = forecaster.forecast(
        current_glucose=current_g,
        current_roc=current_roc,
        iob=twin.current_iob,
        isf=twin.insulin_sensitivity_factor,
        horizon_minutes=horizon_minutes
    )

    return {
        "patient_id": patient_id,
        "horizon_minutes": horizon_minutes,
        "trajectory": steps
    }


@router.get("/{patient_id}/shap", dependencies=[Depends(verify_api_key)])
async def get_patient_shap(patient_id: str):
    """Retrieve top SHAP explanations for recent critical prediction."""
    engine = get_twin_engine()
    twin = engine.get_twin(patient_id)
    if twin is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient Twin '{patient_id}' not found."
        )

    if twin.last_alert and twin.last_alert.tier == "RED" and twin.last_alert.shap_features:
        return {
            "patient_id": patient_id,
            "timestamp": twin.last_alert.timestamp.isoformat(),
            "explanations": twin.last_alert.shap_features
        }

    return {"message": "No RED alert found for this patient"}


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_api_key)])
async def delete_patient(patient_id: str):
    """Delete and decommission Patient Twin."""
    engine = get_twin_engine()
    if not engine.remove_twin(patient_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient Twin '{patient_id}' not found."
        )
    return None
