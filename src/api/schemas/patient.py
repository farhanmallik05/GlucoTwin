"""
src/api/schemas/patient.py
Pydantic schemas for Patient Twin registration, telemetry updates, and predictions.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

RiskTier = Literal["GREEN", "YELLOW", "ORANGE", "RED"]
TrendDirection = Literal["rapidly_rising", "rising", "stable", "falling", "rapidly_falling"]


class PatientCreateRequest(BaseModel):
    patient_id: str = Field(..., description="Unique clinical patient identifier")
    ehr_features: Dict[str, Any] = Field(..., description="Baseline static EHR profile and clinical indicators")


class PatientResponse(BaseModel):
    patient_id: str
    status: str = "created"
    ehr_features: Optional[Dict[str, Any]] = None


class TelemetryUpdateRequest(BaseModel):
    patient_id: Optional[str] = None
    timestamp: Optional[str] = None
    glucose_mgdl: float = Field(..., description="Real-time interstitial glucose reading (mg/dL)")
    glucose_roc: Optional[float] = Field(0.0, description="Rate of change (mg/dL/min)")
    glucose_acc: Optional[float] = 0.0
    heart_rate_bpm: Optional[float] = 72.0
    hrv_rmssd_ms: Optional[float] = Field(35.0, description="Root mean square of successive RR interval differences")
    hrv_sdnn_ms: Optional[float] = 45.0
    hrv_lf_hf_ratio: Optional[float] = 1.2
    steps_count: Optional[int] = 0
    steps_last_5min: Optional[int] = 0
    activity_type: Optional[str] = "resting"
    sleep_stage: Optional[str] = "none"
    sleep_quality: Optional[float] = 0.85


class ForecastStep(BaseModel):
    step: int
    minutes_ahead: int
    glucose_projected: float
    glucose_lower: float
    glucose_upper: float


class ForecastResponse(BaseModel):
    patient_id: str
    horizon_minutes: int = 120
    trajectory: List[ForecastStep]


class PredictionResponse(BaseModel):
    patient_id: str
    timestamp: str
    glucose_current: float
    glucose_trend: TrendDirection
    risk_score: float
    risk_tier: RiskTier
    confidence_lower: float
    confidence_upper: float
    alert_fired: bool
    top_features: Optional[List[Dict[str, Any]]] = None
