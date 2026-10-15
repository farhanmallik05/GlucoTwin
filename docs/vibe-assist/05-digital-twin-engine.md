# Vibe Assist 05 — Digital Twin Engine
## PatientTwin State Machine & Physiological Modelling

> **Vibe Coding Session 05/10** | Estimated time: 4-5 hours
> Goal: The core PatientTwin class — a stateful, updateable virtual patient model.

---

## 🎯 Session Goal

By the end of this session you will have:
- ✅ `PatientTwin` class — full stateful virtual patient
- ✅ `TwinEngine` — manages multiple concurrent twins
- ✅ Insulin-on-board (IOB) pharmacokinetic model
- ✅ Alert system with 4-tier severity
- ✅ Personal baseline calibration from EHR history
- ✅ Twin state persistence (SQLite)

---

## Concept: What Makes a Digital Twin Different

```
Simple Predictor:                 Digital Twin:
┌──────────────┐                  ┌──────────────────────────────────┐
│ Input: X     │                  │  Patient State (persisted)        │
│ Output: Y    │                  │  ┌────────────────────────────┐  │
│ Stateless    │    vs            │  │ glucose_history (48 steps) │  │
└──────────────┘                  │  │ hrv_baseline (personal)    │  │
                                  │  │ insulin_on_board (PK model)│  │
                                  │  │ circadian_phase            │  │
                                  │  │ meal_event_history         │  │
                                  │  └────────────────────────────┘  │
                                  │                                    │
                                  │  Update(new_reading) →            │
                                  │    1. Update state                │
                                  │    2. Recompute features          │
                                  │    3. Run model                   │
                                  │    4. Update risk                 │
                                  │    5. Check alerts                │
                                  │    6. Return PredictionResult     │
                                  └──────────────────────────────────┘

The Twin REMEMBERS context. It knows this patient's normal.
It detects deviations from THEIR baseline, not population averages.
```

---

## 🤖 Vibe Prompt 1 — PatientTwin Core Class

```
Create src/twin/patient_twin.py with the PatientTwin class.

This is the CORE component of GlucoTwin. It represents a single 
patient's digital twin — their stateful virtual counterpart.

```python
from dataclasses import dataclass, field
from typing import Optional, List, Deque, Literal
from collections import deque
from datetime import datetime, timedelta
import numpy as np

# Enums
TrendDirection = Literal["rapidly_rising", "rising", "stable", "falling", "rapidly_falling"]
RiskTier = Literal["GREEN", "YELLOW", "ORANGE", "RED"]
EventType = Literal["hyperglycaemia", "hypoglycaemia", "none"]

@dataclass
class Alert:
    patient_id: str
    timestamp: datetime
    tier: RiskTier
    risk_score: float
    event_type: EventType
    predicted_event_in_minutes: Optional[int]
    message: str
    shap_features: Optional[List[dict]] = None  # populated for RED alerts

@dataclass 
class PredictionResult:
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
    alert: Optional[Alert]
    top_features: Optional[List[dict]]  # SHAP top 10

class PatientTwin:
    """
    Stateful Digital Twin for a Type 2 Diabetes patient.
    Maintains physiological state and processes streaming sensor data.
    """

    def __init__(self, patient_id: str, ehr_features: dict,
                 history_hours: int = 24):
        self.patient_id = patient_id
        self.ehr_features = ehr_features
        self.twin_created_at = datetime.utcnow()
        self.twin_last_updated = datetime.utcnow()

        # Sliding window buffers (48 readings = 4 hours at 5-min intervals)
        steps_4h = 48
        steps_24h = 288
        self.glucose_history: Deque[float] = deque(maxlen=steps_24h)
        self.hrv_history: Deque[float] = deque(maxlen=steps_24h)
        self.activity_history: Deque[str] = deque(maxlen=steps_24h)
        self.roc_history: Deque[float] = deque(maxlen=steps_24h)

        # Current state
        self.current_glucose: float = 0.0
        self.current_hrv: float = 0.0
        self.current_trend: TrendDirection = "stable"
        self.current_risk_score: float = 0.0
        self.current_risk_tier: RiskTier = "GREEN"

        # Personalized baselines (calibrated from EHR + first N readings)
        self.is_calibrated: bool = False
        self.glucose_baseline_mean: float = self._estimate_baseline_glucose()
        self.glucose_baseline_std: float = 25.0  # Will be updated
        self.hrv_baseline_mean: float = 35.0    # Will be updated
        self.insulin_sensitivity_factor: float = self._estimate_isf()

        # IOB (Insulin on Board) state
        self.last_insulin_dose: float = 0.0
        self.last_insulin_time: Optional[datetime] = None
        self.current_iob: float = 0.0

        # Alert history
        self.alert_history: List[Alert] = []
        self.last_alert: Optional[Alert] = None

        # Model references (injected by TwinEngine)
        self._xgb_model = None
        self._lstm_model = None
        self._shap_explainer = None

    def _estimate_baseline_glucose(self) -> float:
        """Estimate personal glucose baseline from HbA1c."""
        # eAG formula: eAG (mg/dL) = 28.7 × HbA1c − 46.7
        hba1c = self.ehr_features.get('hba1c_latest', 8.0)
        return 28.7 * hba1c - 46.7

    def _estimate_isf(self) -> float:
        """Estimate insulin sensitivity factor from patient profile."""
        # ISF: how much 1 unit of insulin drops glucose (mg/dL)
        # Rule of 1800: ISF = 1800 / total_daily_dose
        daily_dose = self.ehr_features.get('insulin_dose_units', 30)
        if daily_dose > 0:
            return 1800 / daily_dose
        return 50.0  # Default for non-insulin users

    def update(self, reading: 'WearableReading') -> PredictionResult:
        """
        Process a new wearable reading and update twin state.
        This is called every 5 minutes by TwinEngine.
        """
        # 1. Update history buffers
        self.glucose_history.append(reading.glucose_mgdl)
        self.hrv_history.append(reading.hrv_rmssd_ms)
        self.activity_history.append(reading.activity_type)
        self.roc_history.append(reading.glucose_roc)

        # 2. Update current state
        self.current_glucose = reading.glucose_mgdl
        self.current_hrv = reading.hrv_rmssd_ms
        self.current_trend = self._compute_trend()

        # 3. Update IOB
        self._update_iob(reading.timestamp)

        # 4. Calibrate baselines after 24h of data
        if len(self.glucose_history) >= 288 and not self.is_calibrated:
            self._calibrate_baselines()

        # 5. Run prediction
        result = self.predict(reading.timestamp)

        # 6. Update twin metadata
        self.twin_last_updated = reading.timestamp
        self.current_risk_score = result.risk_score
        self.current_risk_tier = result.risk_tier

        # 7. Fire alert if needed
        if result.alert_fired:
            self.alert_history.append(result.alert)
            self.last_alert = result.alert

        return result

    def predict(self, timestamp: datetime) -> PredictionResult:
        """Run ML ensemble on current twin state."""
        # (implemented in Session 07 — prediction engine)
        raise NotImplementedError("Attach models via TwinEngine.attach_models()")

    def _compute_trend(self) -> TrendDirection:
        """Compute glucose trend from recent ROC history."""
        if len(self.roc_history) < 3:
            return "stable"
        recent_roc = np.mean(list(self.roc_history)[-3:])
        if recent_roc > 3.0:   return "rapidly_rising"
        elif recent_roc > 1.0: return "rising"
        elif recent_roc < -3.0: return "rapidly_falling"
        elif recent_roc < -1.0: return "falling"
        else:                   return "stable"

    def _update_iob(self, now: datetime) -> None:
        """Update insulin-on-board using biexponential PK model."""
        if self.last_insulin_time is None:
            self.current_iob = 0.0
            return
        # ... PK model implemented in iob_model.py

    def _calibrate_baselines(self) -> None:
        """Personalize baselines from first 24h of data."""
        glucose_arr = np.array(self.glucose_history)
        self.glucose_baseline_mean = np.percentile(glucose_arr, 25)
        self.glucose_baseline_std = np.std(glucose_arr)
        hrv_arr = np.array(self.hrv_history)
        self.hrv_baseline_mean = np.percentile(hrv_arr, 75)  # Max HRV = best baseline
        self.is_calibrated = True

    def get_summary(self) -> dict:
        """Return twin state summary for API response."""
        return {
            "patient_id": self.patient_id,
            "last_updated": self.twin_last_updated.isoformat(),
            "is_calibrated": self.is_calibrated,
            "current_glucose": self.current_glucose,
            "trend": self.current_trend,
            "risk_score": self.current_risk_score,
            "risk_tier": self.current_risk_tier,
            "current_iob": self.current_iob,
            "total_alerts_today": sum(
                1 for a in self.alert_history
                if a.timestamp.date() == datetime.utcnow().date()
            ),
        }
```

Implement all methods fully. Add comprehensive docstrings.
Create tests/test_patient_twin.py with tests for:
- update() correctly appends to history buffers
- _compute_trend() returns correct direction
- _calibrate_baselines() sets reasonable values
- get_summary() returns valid dict
```

---

## 🤖 Vibe Prompt 2 — TwinEngine (Multi-Patient Coordinator)

```
Create src/twin/engine.py with TwinEngine class that manages 
multiple PatientTwin instances concurrently.

class TwinEngine:
    """Coordinates multiple PatientTwin instances."""

    def __init__(self, xgb_model, lstm_model, shap_explainer):
        self.twins: Dict[str, PatientTwin] = {}
        self.xgb_model = xgb_model
        self.lstm_model = lstm_model
        self.shap_explainer = shap_explainer
        self.alert_callbacks: List[Callable] = []

    def create_twin(self, patient_id: str, 
                    ehr_features: dict) -> PatientTwin:
        """Create a new patient twin."""
        twin = PatientTwin(patient_id, ehr_features)
        twin._xgb_model = self.xgb_model
        twin._lstm_model = self.lstm_model
        twin._shap_explainer = self.shap_explainer
        self.twins[patient_id] = twin
        return twin

    def get_twin(self, patient_id: str) -> Optional[PatientTwin]:
        return self.twins.get(patient_id)

    async def process_reading(self, reading: WearableReading) -> PredictionResult:
        """Process a single reading for the appropriate twin."""
        twin = self.get_twin(reading.patient_id)
        if not twin:
            raise TwinNotFoundError(f"No twin for patient {reading.patient_id}")
        result = twin.update(reading)
        if result.alert_fired:
            await self._trigger_alert_callbacks(result.alert)
        return result

    async def process_batch(self, 
                            readings: List[WearableReading]) -> List[PredictionResult]:
        """Process multiple readings concurrently."""
        tasks = [self.process_reading(r) for r in readings]
        return await asyncio.gather(*tasks)

    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """Register callback to fire when alerts are generated."""
        self.alert_callbacks.append(callback)

    async def _trigger_alert_callbacks(self, alert: Alert):
        for cb in self.alert_callbacks:
            await cb(alert) if asyncio.iscoroutinefunction(cb) else cb(alert)

    def get_all_twin_summaries(self) -> List[dict]:
        return [twin.get_summary() for twin in self.twins.values()]

    def remove_twin(self, patient_id: str) → bool:
        return bool(self.twins.pop(patient_id, None))

Also implement a global TwinEngine singleton for use by the FastAPI app:
    get_twin_engine() → TwinEngine  (lazy-initialized singleton)
```

---

## 🤖 Vibe Prompt 3 — Insulin-on-Board PK Model

```
Create src/twin/iob_model.py with a pharmacokinetic insulin-on-board model.

The IOB model estimates how much active insulin remains in the body 
after an injection. This is critical for hypoglycaemia risk assessment.

class IOBModel:
    """
    Biexponential pharmacokinetic model for insulin-on-board calculation.
    Based on Hovorka et al. (2004) insulin absorption model.
    """

    # Duration of action by insulin type (hours)
    INSULIN_DURATION = {
        "rapid_acting": 4.0,    # Humalog, Novolog, Apidra
        "short_acting": 6.0,    # Regular insulin
        "long_acting": 24.0,    # Glargine, Detemir (flat profile)
        "intermediate": 16.0,   # NPH
        "default": 4.0,
    }

    def __init__(self, insulin_type: str = "default"):
        self.duration_hours = self.INSULIN_DURATION.get(insulin_type, 4.0)
        self.peak_hours = self.duration_hours * 0.35  # Peak at 35% of duration

    def calculate_iob(self, dose_units: float, 
                      hours_since_injection: float) -> float:
        """
        Calculate insulin-on-board using biexponential decay.
        
        Returns: IOB in units (between 0 and dose_units)
        
        The biexponential model:
        IOB(t) = dose × [A × exp(-t/τ1) + (1-A) × exp(-t/τ2)]
        where τ1, τ2 are time constants from peak and duration.
        """
        if hours_since_injection >= self.duration_hours:
            return 0.0
        if hours_since_injection < 0:
            return dose_units

        t = hours_since_injection
        duration = self.duration_hours
        peak = self.peak_hours

        # Biexponential parameters
        tau1 = peak / np.log(2)
        tau2 = (duration - peak) / np.log(2)

        A = 0.7  # Fast component fraction
        iob_fraction = A * np.exp(-t / tau1) + (1 - A) * np.exp(-t / tau2)
        return dose_units * iob_fraction

    def estimate_glucose_lowering(self, iob_units: float, 
                                   isf: float) -> float:
        """
        Estimate expected glucose drop from remaining IOB.
        
        isf: Insulin Sensitivity Factor (mg/dL per unit)
        Returns: Expected glucose decrease in mg/dL
        """
        return iob_units * isf

    def high_iob_risk(self, iob_units: float, 
                       current_glucose: float, isf: float) -> bool:
        """
        Flag if current IOB could drive glucose to hypoglycaemia.
        Returns True if projected glucose < 80 mg/dL.
        """
        projected = current_glucose - self.estimate_glucose_lowering(iob_units, isf)
        return projected < 80.0

Include unit tests validating:
- IOB starts at dose_units at t=0
- IOB reaches 0 at t=duration
- IOB is always non-negative
- high_iob_risk correctly identifies dangerous combinations
```

---

## 🤖 Vibe Prompt 4 — Alert System

```
Create src/twin/alert_system.py with a tiered alert system.

class AlertSystem:
    """
    Generates tiered clinical alerts based on prediction risk scores.
    """

    TIER_THRESHOLDS = {
        "GREEN":  (0.0, 0.30),
        "YELLOW": (0.30, 0.60),
        "ORANGE": (0.60, 0.80),
        "RED":    (0.80, 1.01),
    }

    ALERT_MESSAGES = {
        "GREEN": "Patient within normal range. No intervention required.",
        "YELLOW": "Elevated risk of glycaemic excursion. Monitor closely.",
        "ORANGE": "High risk of adverse event within 90-120 minutes. Consider intervention.",
        "RED": "Critical: Adverse glycaemic event predicted within 60 minutes. Immediate action required.",
    }

    def classify_risk(self, risk_score: float) -> RiskTier:
        """Map risk score to tier."""
        for tier, (low, high) in self.TIER_THRESHOLDS.items():
            if low <= risk_score < high:
                return tier
        return "RED"

    def generate_alert(self, patient_id: str, timestamp: datetime,
                       risk_score: float, glucose: float, 
                       glucose_roc: float, iob: float,
                       shap_features: Optional[List] = None) → Optional[Alert]:
        """
        Generate an Alert only if tier >= ORANGE (avoids alert fatigue).
        Returns None for GREEN and YELLOW tiers.
        """
        tier = self.classify_risk(risk_score)
        if tier in ("GREEN", "YELLOW"):
            return None

        event_type = self._predict_event_type(glucose, glucose_roc, iob)
        eta = self._estimate_event_eta(risk_score, glucose, glucose_roc)

        message = self._build_clinical_message(tier, event_type, eta, 
                                                glucose, iob, shap_features)
        return Alert(
            patient_id=patient_id,
            timestamp=timestamp,
            tier=tier,
            risk_score=risk_score,
            event_type=event_type,
            predicted_event_in_minutes=eta,
            message=message,
            shap_features=shap_features if tier == "RED" else None,
        )

    def _predict_event_type(self, glucose: float, 
                             roc: float, iob: float) → EventType:
        """Determine if event is likely hypo or hyper."""
        if glucose < 100 and roc < -1.0:
            return "hypoglycaemia"
        elif glucose < 100 and iob > 3.0:
            return "hypoglycaemia"
        elif glucose > 200 and roc > 1.0:
            return "hyperglycaemia"
        elif glucose > 180:
            return "hyperglycaemia"
        else:
            return "hyperglycaemia"  # Default to hyper for ambiguous cases

    def _estimate_event_eta(self, risk_score: float, 
                             glucose: float, roc: float) → int:
        """Estimate minutes until event based on risk score and trend."""
        # Higher risk = sooner event
        base_eta = int((1 - risk_score) * 120)  # 0 to 120 minutes
        # Adjust for trend
        if abs(roc) > 3.0:  # Rapid change
            base_eta = int(base_eta * 0.7)
        return max(10, min(120, base_eta))

    def _build_clinical_message(self, tier, event_type, eta, 
                                 glucose, iob, shap) → str:
        """Build human-readable clinical message."""
        base = self.ALERT_MESSAGES[tier]
        details = f" | Current glucose: {glucose:.0f} mg/dL"
        if iob > 1.0:
            details += f" | IOB: {iob:.1f} units"
        if event_type == "hypoglycaemia":
            details += " | Consider: carbohydrate intake or insulin dose reduction"
        elif event_type == "hyperglycaemia":
            details += " | Consider: correction dose or dietary review"
        return base + details

    def check_alert_fatigue(self, alert_history: List[Alert], 
                             new_alert: Alert) → bool:
        """
        Suppress alert if same tier alert was sent within last 30 minutes.
        Prevents alert fatigue — a critical clinical design principle.
        """
        cutoff = new_alert.timestamp - timedelta(minutes=30)
        recent = [a for a in alert_history if a.timestamp > cutoff]
        return any(a.tier == new_alert.tier for a in recent)

Full docstrings, type hints, and tests for all methods.
```

---

## 🤖 Vibe Prompt 5 — Twin State Persistence

```
Create src/twin/persistence.py for saving and loading PatientTwin state.

class TwinPersistence:
    """SQLite-backed persistence for PatientTwin state."""

    def __init__(self, db_path: str = "glucotwin.db"):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        """Create tables if not exist."""
        # Tables needed:
        # 1. twin_state: patient_id, json_state, last_updated
        # 2. prediction_log: patient_id, timestamp, risk_score, 
        #                    risk_tier, glucose, alert_fired
        # 3. alert_log: patient_id, timestamp, tier, risk_score,
        #               event_type, message, shap_json

    def save_twin_state(self, twin: PatientTwin) → None:
        """Serialize and save twin state to DB."""
        state = {
            "glucose_history": list(twin.glucose_history),
            "hrv_history": list(twin.hrv_history),
            "activity_history": list(twin.activity_history),
            "is_calibrated": twin.is_calibrated,
            "glucose_baseline_mean": twin.glucose_baseline_mean,
            "glucose_baseline_std": twin.glucose_baseline_std,
            "hrv_baseline_mean": twin.hrv_baseline_mean,
            "current_iob": twin.current_iob,
            "last_insulin_dose": twin.last_insulin_dose,
        }
        # Save as JSON to twin_state table

    def load_twin_state(self, patient_id: str, 
                         ehr_features: dict) → Optional[PatientTwin]:
        """Restore PatientTwin from persisted state."""

    def log_prediction(self, result: PredictionResult) → None:
        """Log every prediction to prediction_log table."""

    def log_alert(self, alert: Alert) → None:
        """Log every fired alert to alert_log table."""

    def get_prediction_history(self, patient_id: str, 
                                hours: int = 24) → List[dict]:
        """Get recent prediction history for dashboard charts."""

    def get_alert_history(self, patient_id: str) → List[Alert]:
        """Get all alerts for a patient."""

Use SQLAlchemy Core (not ORM) for simplicity.
Include async variants using aiosqlite for the FastAPI routes.
```

---

## 📋 Session 05 Completion Checklist

- [ ] `PatientTwin` class complete with all methods
- [ ] `TwinEngine` manages multiple twins concurrently
- [ ] `IOBModel` correctly computes biexponential IOB decay
- [ ] `AlertSystem` generates tiered alerts with fatigue suppression
- [ ] `TwinPersistence` saves/loads twin state from SQLite
- [ ] All unit tests pass (> 20 tests total)
- [ ] Twin update cycle completes in < 50ms (before model inference)

## 🔜 Next Session

→ **06-ml-model-training.md** — XGBoost + LSTM training and evaluation

---

*GlucoTwin Vibe Assist 05 | Digital Twin Engine | Digital Twin Challenge 2026*
