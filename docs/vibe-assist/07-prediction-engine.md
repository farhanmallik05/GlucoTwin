# Vibe Assist 07 — Prediction Engine
## Wiring PatientTwin.predict() + Real-Time Inference

> **Vibe Coding Session 07/10** | Estimated time: 3 hours
> Goal: Complete PatientTwin.predict() method integrating ensemble model + SHAP, achieving < 200ms end-to-end latency.

---

## 🎯 Session Goal

By the end of this session you will have:
- ✅ `PatientTwin.predict()` fully implemented
- ✅ Feature vector construction from live twin state
- ✅ LSTM sequence construction from history buffers
- ✅ Parallel inference (XGBoost + LSTM run simultaneously)
- ✅ SHAP triggered only on RED alerts (performance optimization)
- ✅ Latency profiled and confirmed < 200ms

---

## Prediction Flow Architecture

```
PatientTwin.update(reading) called
         │
         ▼
   Update history buffers
         │
         ▼
   PatientTwin.predict(timestamp)
         │
         ├──────────────────────────────────────────┐
         │                                          │
         ▼                                          ▼
   Build tabular features (89)         Build LSTM sequence (48×3)
   from ehr_features + history         from glucose/hrv/activity history
   → pd.DataFrame (1 row)             → torch.Tensor (1, 48, 3)
         │                                          │
         ▼ (parallel via asyncio.gather)            ▼
   xgb.predict_proba()              lstm.predict_proba()
   ~15ms                            ~80ms
         │                                          │
         └──────────────┬───────────────────────────┘
                        │
                        ▼
              ensemble.predict()
              Adaptive weights → risk_score
                        │
                        ▼
              alert_system.classify_risk()
              → risk_tier (GREEN/YELLOW/ORANGE/RED)
                        │
                        ├── if RED: shap.explain_prediction() ~50ms
                        │
                        ▼
              Build PredictionResult
                        │
                        ▼
              persistence.log_prediction()
                        │
                        ▼
              Return PredictionResult to TwinEngine
```

---

## 🤖 Vibe Prompt 1 — Complete PatientTwin.predict() Implementation

```
Implement the PatientTwin.predict() method from patient_twin.py.

This method must:
1. Build the 89-feature tabular vector from twin state
2. Build the (48, 3) LSTM sequence from history buffers
3. Run XGBoost and LSTM predictions (use concurrent.futures for parallel)
4. Ensemble predictions with adaptive weights
5. Generate SHAP explanation if risk_tier == RED
6. Return a complete PredictionResult object

Here is the complete implementation:

def predict(self, timestamp: datetime) → PredictionResult:
    """
    Run ML ensemble on current twin state to predict adverse event risk.
    
    Returns:
        PredictionResult with risk score, tier, alert, and SHAP features.
    
    Performance target: < 200ms total (excluding model loading)
    """
    # Guard: need enough history
    if len(self.glucose_history) < 12:  # Need at least 1 hour
        return self._insufficient_data_result(timestamp)

    # 1. Build tabular feature vector (89 features)
    feature_dict = self._build_tabular_features()
    X_tabular = pd.DataFrame([feature_dict])

    # 2. Build LSTM sequence (48 timesteps × 3 channels)
    X_sequence = self._build_lstm_sequence()

    # 3. Run models in parallel (using concurrent.futures)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        xgb_future = executor.submit(
            self._xgb_model.predict_proba, X_tabular
        )
        lstm_future = executor.submit(
            self._lstm_model.predict_proba_sequence, X_sequence
        )
        xgb_prob = float(xgb_future.result()[0])
        lstm_prob = float(lstm_future.result()[0])

    # 4. Ensemble
    risk_score = self._ensemble.predict(xgb_prob, lstm_prob,
                                         self.ehr_features)
    risk_tier = self._alert_system.classify_risk(risk_score)

    # 5. Confidence interval via MC Dropout (fast: 5 passes)
    ci_lower, ci_upper = self._compute_confidence_interval(X_sequence, n=5)

    # 6. SHAP (only for RED alerts — performance optimization)
    shap_features = None
    if risk_tier == "RED" and self._shap_explainer is not None:
        shap_features = self._shap_explainer.explain_prediction(X_tabular)

    # 7. Generate alert
    alert = self._alert_system.generate_alert(
        patient_id=self.patient_id,
        timestamp=timestamp,
        risk_score=risk_score,
        glucose=self.current_glucose,
        glucose_roc=list(self.roc_history)[-1] if self.roc_history else 0.0,
        iob=self.current_iob,
        shap_features=shap_features,
    )
    # Alert fatigue check
    if alert and self._alert_system.check_alert_fatigue(self.alert_history, alert):
        alert = None

    return PredictionResult(
        patient_id=self.patient_id,
        timestamp=timestamp,
        glucose_current=self.current_glucose,
        glucose_trend=self.current_trend,
        risk_score=risk_score,
        risk_tier=risk_tier,
        predicted_event_in_minutes=(alert.predicted_event_in_minutes 
                                    if alert else None),
        confidence_lower=ci_lower,
        confidence_upper=ci_upper,
        alert_fired=alert is not None,
        alert=alert,
        top_features=shap_features[:5] if shap_features else None,
    )

Now implement _build_tabular_features() and _build_lstm_sequence():

def _build_tabular_features(self) → dict:
    """
    Compute 89 features from twin state for XGBoost input.
    Uses pre-computed history buffers.
    """
    from src.data.preprocessing.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    
    # Convert history deques to DataFrame
    n = min(288, len(self.glucose_history))
    window_df = pd.DataFrame({
        "glucose_mgdl": list(self.glucose_history)[-n:],
        "glucose_roc": list(self.roc_history)[-n:],
        "hrv_rmssd_ms": list(self.hrv_history)[-n:],
        "activity_type": list(self.activity_history)[-n:],
        # ... other channels
    })
    return fe.compute_features(window_df, self.ehr_features)

def _build_lstm_sequence(self) → torch.Tensor:
    """
    Build (1, 48, 3) tensor for LSTM input.
    3 channels: normalized glucose, normalized HRV, activity_encoded
    """
    n = min(48, len(self.glucose_history))
    glucose = np.array(list(self.glucose_history)[-n:])
    hrv = np.array(list(self.hrv_history)[-n:])
    activity = np.array([self._encode_activity(a) 
                          for a in list(self.activity_history)[-n:]])
    
    # Pad if less than 48 steps
    if n < 48:
        pad = 48 - n
        glucose = np.pad(glucose, (pad, 0), mode="edge")
        hrv = np.pad(hrv, (pad, 0), mode="edge")
        activity = np.pad(activity, (pad, 0), mode="constant")
    
    # Normalize (use patient baseline, not population)
    glucose_norm = (glucose - self.glucose_baseline_mean) / self.glucose_baseline_std
    hrv_norm = (hrv - self.hrv_baseline_mean) / (self.hrv_baseline_mean * 0.3 + 1e-6)
    
    # Stack channels: (48, 3)
    sequence = np.stack([glucose_norm, hrv_norm, activity], axis=1)
    return torch.FloatTensor(sequence).unsqueeze(0)  # (1, 48, 3)

Implement all helper methods. Add performance timing logs (DEBUG level).
```

---

## 🤖 Vibe Prompt 2 — Performance Profiling

```
Create scripts/profile_prediction.py to measure prediction latency.

I need to confirm end-to-end prediction latency < 200ms per patient.

import time
import cProfile
from src.twin.engine import get_twin_engine
from src.data.ingestion.wearable_stream import WearableReading

def benchmark_prediction(n_patients: int = 10, n_iterations: int = 100):
    engine = get_twin_engine()
    
    # Create N test patients
    for i in range(n_patients):
        engine.create_twin(f"BENCH-{i:03d}", sample_ehr_features())
    
    # Pre-populate 4 hours of history for each twin
    for twin in engine.twins.values():
        for _ in range(48):
            twin.glucose_history.append(np.random.normal(140, 20))
            twin.hrv_history.append(np.random.normal(35, 10))
            twin.activity_history.append("resting")
            twin.roc_history.append(np.random.normal(0, 0.5))
    
    # Benchmark
    latencies = []
    for i in range(n_iterations):
        reading = sample_wearable_reading("BENCH-000")
        start = time.perf_counter()
        result = engine.process_reading_sync(reading)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
    
    print(f"\n=== Prediction Latency Benchmark ===")
    print(f"N iterations: {n_iterations}")
    print(f"Mean:   {np.mean(latencies):.1f}ms")
    print(f"P50:    {np.percentile(latencies, 50):.1f}ms")
    print(f"P95:    {np.percentile(latencies, 95):.1f}ms")
    print(f"P99:    {np.percentile(latencies, 99):.1f}ms")
    print(f"Max:    {np.max(latencies):.1f}ms")
    
    assert np.percentile(latencies, 95) < 200, \
        f"❌ P95 latency {np.percentile(latencies, 95):.1f}ms exceeds 200ms target!"
    print(f"\n✅ Latency target met: P95 < 200ms")
    
    return latencies

if __name__ == "__main__":
    benchmark_prediction()

Also create a cProfile-based analysis:
    python -m cProfile -o profile.out scripts/profile_prediction.py
    python -c "import pstats; p = pstats.Stats('profile.out'); p.sort_stats('cumulative'); p.print_stats(20)"

Expected optimization targets if latency > 200ms:
- Cache FeatureEngineer instance (don't create per prediction)
- Pre-allocate torch tensors (avoid repeated allocation)
- Use numpy arrays for history buffers (faster than deques for slicing)
- Batch XGBoost DMatrix creation
```

---

## 🤖 Vibe Prompt 3 — Prediction History + Time Series Output

```
The React dashboard needs to show the glucose prediction 2 hours into 
the future (as a shaded region on the chart). 

Create src/twin/forecaster.py with GlucoseForecaster:

class GlucoseForecaster:
    """
    Projects glucose trajectory 120 minutes into the future.
    Used to populate the prediction band on the dashboard chart.
    """

    def __init__(self, twin: PatientTwin):
        self.twin = twin

    def forecast_trajectory(self, n_steps: int = 24) → List[dict]:
        """
        Generate 24 forecast points (each = 5 minutes = 2 hours total).
        
        Method: Simple AR(1) model using current glucose + rate of change,
        modulated by IOB model for insulin effect.
        
        This is NOT a deep prediction — it's a visual guide showing the 
        expected trajectory if current trends continue.
        
        Returns list of {timestamp, glucose_projected, confidence_width}
        """
        current_glucose = self.twin.current_glucose
        current_roc = list(self.twin.roc_history)[-1] if self.twin.roc_history else 0.0
        current_iob = self.twin.current_iob
        isf = self.twin.insulin_sensitivity_factor

        forecast = []
        t = datetime.utcnow()
        g = current_glucose
        roc = current_roc

        for step in range(n_steps):
            t += timedelta(minutes=5)
            
            # Decay rate of change (glucose can't keep rising/falling forever)
            roc *= 0.90  # ROC decays by 10% each step

            # IOB effect
            iob_glucose_effect = -isf * (current_iob * 0.05)  # 5% of IOB active per step

            # Project
            g = g + (roc * 5) + iob_glucose_effect
            g = max(40, min(400, g))  # Clip to physiological range

            # Uncertainty grows with time
            confidence_width = 10 + step * 2  # ±(10 + 2×step) mg/dL

            forecast.append({
                "timestamp": t.isoformat(),
                "glucose_projected": round(g, 1),
                "glucose_lower": round(max(40, g - confidence_width), 1),
                "glucose_upper": round(min(400, g + confidence_width), 1),
                "step": step + 1,
                "minutes_ahead": (step + 1) * 5,
            })

        return forecast

Add this to the PatientTwin class:
    def get_glucose_forecast(self) → List[dict]:
        from src.twin.forecaster import GlucoseForecaster
        return GlucoseForecaster(self).forecast_trajectory()

This forecast is used by the API endpoint:
    GET /api/v1/patients/{id}/forecast
    Response: {
        "patient_id": "...",
        "forecast_generated_at": "...",
        "horizon_minutes": 120,
        "trajectory": [...24 points...]
    }
```

---

## 🤖 Vibe Prompt 4 — Demo Scenario Runner

```
Create scripts/run_demo_scenario.py that runs a pre-scripted demo 
scenario for hackathon demonstrations.

DEMO SCENARIO: "The Post-Lunch Hyperglycaemia Alert"
  - Patient: Rajesh Kumar, 52yr, BMI 27, HbA1c 8.4%, on Metformin + SGLT2
  - Normal morning glucose: 120-145 mg/dL
  - 12:30 PM: Large lunch (biryani + dessert) — unlogged
  - 13:00 PM: Glucose starts rising (ROC +2.5 mg/dL/min)
  - 13:15 PM: HRV starts dropping (sympathetic activation)
  - 13:30 PM: Model hits ORANGE tier (risk 0.65)
  - 13:45 PM: Model hits RED tier (risk 0.82)
  - 14:30 PM: Glucose peaks at 287 mg/dL (excursion confirmed)
  - Without intervention: patient calls ER at 16:00

class DemoScenarioRunner:
    def __init__(self, engine: TwinEngine, 
                 persistence: TwinPersistence,
                 speed_multiplier: float = 60.0):  # 1 min of data per second
        
    def setup_demo_patient(self) → str:
        """Create Rajesh's twin with pre-seeded baseline history."""
        ehr = {
            "patient_id": "DEMO-RAJESH",
            "age": 52, "gender": "M", "bmi": 27.2,
            "hba1c_latest": 8.4, "fasting_glucose_avg": 158,
            "on_metformin": True, "on_insulin": False,
            "on_sglt2": True, "has_hypertension": True,
            "diabetes_duration_years": 6,
            # ... all 23 features
        }
        patient_id = "DEMO-RAJESH"
        self.engine.create_twin(patient_id, ehr)
        self._seed_baseline_history(patient_id)
        return patient_id

    async def run_scenario(self) → AsyncGenerator[PredictionResult, None]:
        """
        Yield PredictionResults as the scenario plays out.
        At demo speed (60×), 2 hours of scenario plays in ~2 minutes.
        """
        # Pre-generated scenario readings
        scenario_readings = self._generate_scenario_readings()
        for reading in scenario_readings:
            result = await self.engine.process_reading(reading)
            await asyncio.sleep(1 / self.speed_multiplier)
            yield result

    def _generate_scenario_readings(self) → List[WearableReading]:
        """Generate the scripted post-lunch hyperglycaemia scenario."""
        # ... 120 readings (5-min intervals over 10 hours)
        # ... including the post-lunch spike and HRV suppression

Run with: python scripts/run_demo_scenario.py
It should print a table showing the 20 most important timesteps:
  13:00 | Glucose 145 | ROC +2.1 | HRV 38ms | Risk 0.28 | GREEN
  13:15 | Glucose 172 | ROC +2.8 | HRV 29ms | Risk 0.51 | YELLOW
  13:30 | Glucose 201 | ROC +3.2 | HRV 22ms | Risk 0.67 | ORANGE ⚠️
  13:45 | Glucose 231 | ROC +2.9 | HRV 19ms | Risk 0.83 | RED 🚨
```

---

## ✅ Prediction Engine Integration Test

```python
# tests/test_prediction_engine.py

import pytest
from src.twin.patient_twin import PatientTwin
from src.data.ingestion.wearable_stream import WearableReading

def test_predict_requires_history():
    twin = PatientTwin("TEST-001", sample_ehr())
    # Empty history → insufficient data result
    result = twin.predict(datetime.utcnow())
    assert result.risk_tier == "GREEN"
    assert result.confidence_lower == 0.0

def test_predict_high_glucose_rising_triggers_orange():
    twin = create_twin_with_rising_glucose(start=160, roc=3.0)
    result = twin.predict(datetime.utcnow())
    assert result.risk_score > 0.5
    assert result.risk_tier in ("ORANGE", "RED")

def test_predict_normal_glucose_stays_green():
    twin = create_twin_with_stable_glucose(glucose=120)
    result = twin.predict(datetime.utcnow())
    assert result.risk_tier == "GREEN"
    assert result.risk_score < 0.30

def test_prediction_latency():
    import time
    twin = create_well_populated_twin()
    start = time.perf_counter()
    twin.predict(datetime.utcnow())
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 200, f"Prediction took {elapsed_ms:.1f}ms (target: <200ms)"

def test_red_alert_generates_shap():
    twin = create_twin_in_crisis_state()
    result = twin.predict(datetime.utcnow())
    if result.risk_tier == "RED":
        assert result.top_features is not None
        assert len(result.top_features) > 0
        assert all("feature" in f and "shap_value" in f 
                   for f in result.top_features)
```

---

## 📋 Session 07 Completion Checklist

- [ ] `PatientTwin.predict()` fully implemented
- [ ] `_build_tabular_features()` returns all 89 features
- [ ] `_build_lstm_sequence()` returns correct (1, 48, 3) tensor
- [ ] Parallel XGBoost + LSTM inference working
- [ ] SHAP triggered only on RED alerts
- [ ] P95 prediction latency < 200ms (benchmark confirmed)
- [ ] `GlucoseForecaster` returns 24-point trajectory
- [ ] Demo scenario script runs and prints prediction table
- [ ] All prediction engine tests pass

## 🔜 Next Session

→ **08-api-backend.md** — FastAPI REST + WebSocket implementation

---

*GlucoTwin Vibe Assist 07 | Prediction Engine | Digital Twin Challenge 2026*
