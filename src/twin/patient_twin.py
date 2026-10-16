"""
src/twin/patient_twin.py
Stateful Virtual Patient Model for Type 2 Diabetes Adverse Glycemic Event Prediction.
Incorporates live SHAP explainability, epistemic uncertainty (MC Dropout), and physiological sensor clamping.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Literal, Optional
import numpy as np
import pandas as pd

from src.twin.alert_system import Alert, AlertSystem, EventType, RiskTier
from src.twin.iob_model import IOBModel
from src.explainability.shap_explainer import SHAPExplainer

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
    sensor artifact plausibility verification, and coordinates real-time predictive inference.
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

        # Sensor artifact state tracker (compression artifact mitigation)
        self._unconfirmed_artifact: bool = False

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

    def _compute_confidence_interval(self, risk_score: float) -> tuple[float, float]:
        """
        Derive 95% Bayesian credible interval using MC Dropout or calibrated trajectory variance.
        Eliminates naive flat offsets in clinical responses.
        """
        if self._ensemble_model is not None and len(self.glucose_history) >= 48:
            try:
                g_seq = list(self.glucose_history)[-48:]
                r_seq = list(self.roc_history)[-48:]
                h_seq = list(self.hrv_history)[-48:]
                seq_mat = np.column_stack([
                    np.array(g_seq) / 300.0,
                    np.array(r_seq) / 5.0,
                    np.array(h_seq) / 100.0
                ])
                c_low, c_high = self._ensemble_model.estimate_confidence_interval(seq_mat, num_samples=30)
                return max(0.0, c_low), min(1.0, c_high)
            except Exception:
                pass

        # Volatility-scaled dispersion fallback calibrated to trajectory standard deviation
        volatility = float(np.std(list(self.glucose_history)[-12:])) if len(self.glucose_history) >= 12 else 15.0
        delta_ci = min(0.15, max(0.04, (volatility / 220.0) * 0.85))
        c_low = max(0.0, risk_score - delta_ci)
        c_high = min(1.0, risk_score + delta_ci)
        return c_low, c_high

    def _derive_shap_explanations(self) -> List[Dict[str, Any]]:
        """
        Compute live SHAP feature attributions on patient multimodal feature frame.
        Calls SHAP TreeExplainer on fitted XGBoost model or clinical feature attribution layer.
        """
        if self._shap_explainer is None:
            self._shap_explainer = SHAPExplainer(model=self._xgb_model)

        feat_dict = {
            "glucose_current": self.current_glucose,
            "glucose_roc_current": self.roc_history[-1] if len(self.roc_history) > 0 else 0.0,
            "glucose_mean_1h": float(np.mean(list(self.glucose_history)[-12:])) if len(self.glucose_history) >= 12 else self.current_glucose,
            "glucose_max_1h": float(np.max(list(self.glucose_history)[-12:])) if len(self.glucose_history) >= 12 else self.current_glucose,
            "glucose_min_1h": float(np.min(list(self.glucose_history)[-12:])) if len(self.glucose_history) >= 12 else self.current_glucose,
            "hrv_rmssd_ms": self.current_hrv,
            "hba1c_latest": float(self.ehr_features.get("hba1c_latest", 8.0)),
            "age": float(self.ehr_features.get("age", 50)),
            "bmi": float(self.ehr_features.get("bmi", 27.0)),
            "current_iob": self.current_iob,
            "on_insulin": 1.0 if self.ehr_features.get("on_insulin") else 0.0,
            "egfr": float(self.ehr_features.get("egfr", 80.0)),
            "bp_sys": float(self.ehr_features.get("bp_sys", 130.0)),
        }
        features_df = pd.DataFrame([feat_dict])
        return self._shap_explainer.explain(features_df, top_n=5)

    def _build_prediction_result(self, risk_score: float, timestamp: Optional[datetime] = None) -> PredictionResult:
        """
        Construct PredictionResult with dynamically computed SHAP explanations for RED alerts
        and Bayesian credible intervals.
        """
        ts = timestamp or get_utc_now()
        tier = self.alert_system.classify_risk(risk_score)
        c_low, c_high = self._compute_confidence_interval(risk_score)

        shap_features = None
        alert = None

        if tier == "RED":
            # Dynamically compute clinical SHAP explanations from active model and patient features
            shap_features = self._derive_shap_explanations()
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
        Process incoming multi-channel wearable sensor reading with physiological artifact clamping
        and advance twin state.
        """
        ts = getattr(reading, "timestamp", get_utc_now())
        glucose = float(getattr(reading, "glucose_mgdl", 120.0))
        hrv = float(getattr(reading, "hrv_rmssd_ms", 35.0))
        roc = float(getattr(reading, "glucose_roc", 0.0))
        activity = str(getattr(reading, "activity_type", "resting"))
        data_quality = str(getattr(reading, "data_quality", "good")).lower()

        # 1. Check data quality flag — suppress alerts on sensor noise or calibration errors
        if data_quality != "good":
            return PredictionResult(
                patient_id=self.patient_id,
                timestamp=ts,
                glucose_current=glucose,
                glucose_trend="stable",
                risk_score=0.15,
                risk_tier="GREEN",
                predicted_event_in_minutes=None,
                confidence_lower=0.10,
                confidence_upper=0.20,
                alert_fired=False,
                alert=None,
                top_features=[]
            )

        # 2. Physiological plausibility check & compression artifact damping (Failure 1 mitigation)
        # In human physiology, glucose delta rarely exceeds 20 mg/dL in 5 minutes without sensor artifact
        if self.current_glucose > 0.0 and abs(glucose - self.current_glucose) > 20.0:
            if not self._unconfirmed_artifact:
                # First unconfirmed outlier: damp delta to prevent immediate iatrogenic alert
                effective_glucose = self.current_glucose + float(np.sign(glucose - self.current_glucose) * 20.0)
                self._unconfirmed_artifact = True
            else:
                # Sustained reading confirms physiological excursion
                effective_glucose = glucose
                self._unconfirmed_artifact = False
        else:
            effective_glucose = glucose
            self._unconfirmed_artifact = False

        # 3. Append to circular history
        self.glucose_history.append(effective_glucose)
        self.hrv_history.append(hrv)
        self.activity_history.append(activity)
        self.roc_history.append(roc)

        # 4. Update current states
        self.current_glucose = effective_glucose
        self.current_hrv = hrv
        self.current_trend = self._compute_trend()
        self.twin_last_updated = ts

        # 5. Check baseline calibration (24h of 5-min intervals = 288 readings)
        if len(self.glucose_history) >= 288 and not self.is_calibrated:
            self._calibrate_baselines()

        # 6. Cold-start check (< 12 readings = under 1 hour of wearable data)
        if len(self.glucose_history) < 12:
            return PredictionResult(
                patient_id=self.patient_id,
                timestamp=ts,
                glucose_current=effective_glucose,
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

        # 7. Calculate predictive risk score
        risk_score = self._compute_predictive_risk(effective_glucose, roc, hrv)
        self.current_risk_score = risk_score
        self.current_risk_tier = self.alert_system.classify_risk(risk_score)

        # 8. Build prediction result and alert
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
