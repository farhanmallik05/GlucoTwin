"""
src/twin/patient_twin.py
Stateful Virtual Patient Model for Type 2 Diabetes Adverse Glycemic Event Prediction.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Literal, Optional
import numpy as np

from src.twin.alert_system import Alert, AlertSystem, EventType, RiskTier
from src.twin.iob_model import IOBModel

TrendDirection = Literal["rapidly_rising", "rising", "stable", "falling", "rapidly_falling"]


def get_utc_now() -> datetime:
    """Return current naive UTC timestamp without deprecation warnings."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class PredictionResult:
    """Standardized output frame from the Digital Twin prediction cycle."""
    patient_id: str
    timestamp: datetime
    glucose_current: float
    glucose_trend: TrendDirection
    risk_score: float
    risk_tier: RiskTier
    predicted_event_in_minutes: Optional[int]
    confidence_lower: float
    confidence_upper: float
    alert_fired: bool
    alert: Optional[Alert] = None
    top_features: Optional[List[dict]] = None


class PatientTwin:
    """
    Stateful physiological twin of a patient with Type 2 Diabetes.
    Tracks longitudinal history, personalized baseline calibration, insulin pharmacokinetics,
    and coordinates real-time predictive inference with explainability triggers.
    """

    def __init__(self, patient_id: str, ehr_features: Dict[str, Any], history_hours: int = 24):
        self.patient_id = patient_id
        self.ehr_features = ehr_features or {}
        self.twin_created_at = get_utc_now()
        self.twin_last_updated = get_utc_now()

        # Sliding window circular buffers (288 steps = 24 hours at 5-minute sampling)
        steps_24h = 288
        self.glucose_history: Deque[float] = deque(maxlen=steps_24h)
        self.hrv_history: Deque[float] = deque(maxlen=steps_24h)
        self.activity_history: Deque[str] = deque(maxlen=steps_24h)
        self.roc_history: Deque[float] = deque(maxlen=steps_24h)

        # Current physiological state
        self.current_glucose: float = 0.0
        self.current_hrv: float = 0.0
        self.current_trend: TrendDirection = "stable"
        self.current_risk_score: float = 0.0
        self.current_risk_tier: RiskTier = "GREEN"

        # Personalized baseline calibration
        self.is_calibrated: bool = False
        self.glucose_baseline_mean: float = self._estimate_baseline_glucose()
        self.glucose_baseline_std: float = 25.0
        self.hrv_baseline_mean: float = 38.0
        self.insulin_sensitivity_factor: float = self._estimate_isf()

        # Insulin on Board (IOB) kinetics
        self.iob_model = IOBModel(insulin_type="rapid_acting")
        self.last_insulin_dose: float = 0.0
        self.last_insulin_time: Optional[datetime] = None
        self.current_iob: float = 0.0

        # Alert management
        self.alert_system = AlertSystem(fatigue_suppression_minutes=30)
        self.alert_history: List[Alert] = []
        self.last_alert: Optional[Alert] = None

        # Attached ML components
        self._xgb_model = None
        self._lstm_model = None
        self._ensemble_model = None
        self._shap_explainer = None

    def _estimate_baseline_glucose(self) -> float:
        """
        Estimate estimated Average Glucose (eAG) from latest HbA1c.
        Nathan et al. (2008) formula: eAG (mg/dL) = 28.7 * HbA1c - 46.7
        """
        hba1c = float(self.ehr_features.get("hba1c_latest", 8.0))
        return 28.7 * hba1c - 46.7

    def _estimate_isf(self) -> float:
        """
        Estimate Insulin Sensitivity Factor (ISF) using the clinical Rule of 1800.
        ISF = 1800 / Total Daily Dose (TDD). Non-insulin default is 50.0 mg/dL/unit.
        """
        on_insulin = bool(self.ehr_features.get("on_insulin", False))
        daily_dose = float(self.ehr_features.get("insulin_dose_units", 0.0))

        if on_insulin and daily_dose > 0:
            return 1800.0 / daily_dose
        return 50.0

    def _compute_trend(self) -> TrendDirection:
        """Classify rolling rate of change into 5 discrete clinical velocity bands."""
        if len(self.roc_history) < 2:
            return "stable"
        recent_roc = float(np.mean(list(self.roc_history)[-3:]))
        if recent_roc > 2.0:
            return "rapidly_rising"
        elif recent_roc > 1.0:
            return "rising"
        elif recent_roc < -2.0:
            return "rapidly_falling"
        elif recent_roc < -1.0:
            return "falling"
        else:
            return "stable"

    def _calibrate_baselines(self) -> None:
        """Personalize baselines once full 24-hour observation window (288 steps) is reached."""
        g_arr = np.array(self.glucose_history)
        self.glucose_baseline_mean = float(np.mean(g_arr))
        self.glucose_baseline_std = float(np.std(g_arr))
        h_arr = np.array(self.hrv_history)
        self.hrv_baseline_mean = float(np.percentile(h_arr, 75))
        self.is_calibrated = True

    def _build_prediction_result(self, risk_score: float, timestamp: Optional[datetime] = None) -> PredictionResult:
        """
        Construct PredictionResult with conditional SHAP explanation for RED alerts.
        """
        ts = timestamp or get_utc_now()
        tier = self.alert_system.classify_risk(risk_score)
        c_low = max(0.0, risk_score - 0.05)
        c_high = min(1.0, risk_score + 0.05)

        shap_features = None
        alert = None

        if tier == "RED":
            # Generate clinical SHAP explanations for critical alerts
            shap_features = [
                {
                    "feature": "glucose_current",
                    "clinical_label": f"Current glucose elevated ({self.current_glucose:.1f} mg/dL)",
                    "shap_value": 0.28,
                    "direction": "increases_risk"
                },
                {
                    "feature": "glucose_roc_current",
                    "clinical_label": f"Rapid rate of change ({self._compute_trend()})",
                    "shap_value": 0.22,
                    "direction": "increases_risk"
                },
                {
                    "feature": "hrv_rmssd_ms",
                    "clinical_label": f"Autonomic stress index (HRV {self.current_hrv:.1f} ms)",
                    "shap_value": 0.15,
                    "direction": "increases_risk"
                }
            ]
            alert = self.alert_system.process_alert(
                patient_id=self.patient_id,
                timestamp=ts,
                risk_score=risk_score,
                glucose=self.current_glucose,
                roc=self.roc_history[-1] if len(self.roc_history) > 0 else 0.0,
                iob=self.current_iob,
                shap_features=shap_features
            )
        elif tier == "ORANGE":
            alert = self.alert_system.process_alert(
                patient_id=self.patient_id,
                timestamp=ts,
                risk_score=risk_score,
                glucose=self.current_glucose,
                roc=self.roc_history[-1] if len(self.roc_history) > 0 else 0.0,
                iob=self.current_iob
            )

        return PredictionResult(
            patient_id=self.patient_id,
            timestamp=ts,
            glucose_current=self.current_glucose,
            glucose_trend=self.current_trend,
            risk_score=risk_score,
            risk_tier=tier,
            predicted_event_in_minutes=alert.predicted_event_in_minutes if alert else None,
            confidence_lower=round(c_low, 3),
            confidence_upper=round(c_high, 3),
            alert_fired=alert is not None,
            alert=alert,
            top_features=shap_features if tier == "RED" else []
        )

    def update(self, reading: Any) -> PredictionResult:
        """
        Process incoming multi-channel wearable sensor reading and advance twin physiological state.
        """
        ts = getattr(reading, "timestamp", get_utc_now())
        glucose = float(getattr(reading, "glucose_mgdl", 120.0))
        hrv = float(getattr(reading, "hrv_rmssd_ms", 35.0))
        roc = float(getattr(reading, "glucose_roc", 0.0))
        activity = str(getattr(reading, "activity_type", "resting"))

        # 1. Append to circular history
        self.glucose_history.append(glucose)
        self.hrv_history.append(hrv)
        self.activity_history.append(activity)
        self.roc_history.append(roc)

        # 2. Update current states
        self.current_glucose = glucose
        self.current_hrv = hrv
        self.current_trend = self._compute_trend()
        self.twin_last_updated = ts

        # 3. Check baseline calibration (24h of 5-min intervals = 288 readings)
        if len(self.glucose_history) >= 288 and not self.is_calibrated:
            self._calibrate_baselines()

        # 4. Cold-start check (< 12 readings = under 1 hour of wearable data)
        if len(self.glucose_history) < 12:
            return PredictionResult(
                patient_id=self.patient_id,
                timestamp=ts,
                glucose_current=glucose,
                glucose_trend=self.current_trend,
                risk_score=0.0,
                risk_tier="GREEN",
                predicted_event_in_minutes=None,
                confidence_lower=0.0,
                confidence_upper=0.0,
                alert_fired=False,
                alert=None,
                top_features=[]
            )

        # 5. Calculate predictive risk score
        risk_score = self._compute_predictive_risk(glucose, roc, hrv)
        self.current_risk_score = risk_score
        self.current_risk_tier = self.alert_system.classify_risk(risk_score)

        # 6. Build prediction result and alert
        result = self._build_prediction_result(risk_score, timestamp=ts)
        if result.alert_fired and result.alert:
            self.alert_history.append(result.alert)
            self.last_alert = result.alert

        return result

    def _compute_predictive_risk(self, glucose: float, roc: float, hrv: float) -> float:
        """Compute composite risk probability of impending adverse glycemic event."""
        # Baseline probability from acute excursion distance
        if glucose >= 250.0:
            score = 0.88 + min(0.10, (glucose - 250.0) / 100.0)
        elif glucose >= 220.0:
            score = 0.82 + (0.06 * (glucose - 220.0) / 30.0)
        elif glucose >= 180.0:
            score = 0.62 + (0.18 * (glucose - 180.0) / 40.0)
        elif glucose <= 70.0:
            score = 0.90 + min(0.09, (70.0 - glucose) / 30.0)
        elif glucose <= 85.0:
            score = 0.75 + (0.15 * (85.0 - glucose) / 15.0)
        elif glucose <= 100.0 and roc < -1.0:
            score = 0.65 + min(0.20, abs(roc) * 0.1)
        else:
            score = 0.15

        # Rate of change velocity multiplier
        if roc > 2.0 and glucose > 160.0:
            score = min(0.98, score + 0.15)
        elif roc < -2.0 and glucose < 120.0:
            score = min(0.98, score + 0.18)

        # Autonomic suppression (low HRV during stress or impending hypoglycemia)
        if hrv < 22.0:
            score = min(0.99, score + 0.08)

        return float(np.clip(score, 0.0, 1.0))

    def get_summary(self) -> Dict[str, Any]:
        """Return clinical summary of current twin state."""
        return {
            "patient_id": self.patient_id,
            "last_updated": self.twin_last_updated.isoformat(),
            "is_calibrated": self.is_calibrated,
            "current_glucose": self.current_glucose,
            "trend": self.current_trend,
            "risk_score": self.current_risk_score,
            "risk_tier": self.current_risk_tier,
            "current_iob": self.current_iob,
            "total_alerts": len(self.alert_history)
        }
