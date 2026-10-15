# Vibe Assist 04 — Wearable Integration
## Real-Time Sensor Stream Simulation & Processing

> **Vibe Coding Session 04/10** | Estimated time: 3 hours
> Goal: Realistic multi-modal wearable stream that feeds the Digital Twin in real-time.

---

## 🎯 Session Goal

By the end of this session you will have:
- ✅ `WearableStreamManager` — manages real-time sensor streams per patient
- ✅ CGM simulator with noise, lag, and calibration drift (realistic!)
- ✅ HRV computation from raw R-R interval simulation
- ✅ Ohio T1DM dataset adapter (real CGM data when available)
- ✅ Stream multiplexer for demo mode (accelerated playback)
- ✅ Wearable EDA notebook (02_wearable_eda.ipynb)

---

## Clinical Background: Why Each Sensor Matters

```
┌─────────────────────────────────────────────────────────────────┐
│ SENSOR          │ CLINICAL ROLE IN T2D PREDICTION               │
├─────────────────────────────────────────────────────────────────┤
│ CGM             │ PRIMARY — Direct glucose measurement           │
│                 │ Rate of change is more predictive than value   │
├─────────────────────────────────────────────────────────────────┤
│ HRV (RMSSD)     │ SECONDARY — Autonomic nervous system marker   │
│                 │ Drops 30-50% before hypoglycaemia events      │
│                 │ Elevated sympathetic tone precedes hyper events│
├─────────────────────────────────────────────────────────────────┤
│ Step Count      │ TERTIARY — Exercise effect on glucose          │
│                 │ Exercise depletes glucose → hypoglycaemia risk │
│                 │ Sedentary → hyperglycaemia accumulation        │
├─────────────────────────────────────────────────────────────────┤
│ Sleep Stages    │ QUATERNARY — Circadian glucose regulation      │
│                 │ Poor sleep → cortisol → dawn phenomenon        │
│                 │ Deep sleep suppression correlates with HbA1c   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Vibe Prompt 1 — Realistic CGM Simulator with Noise

```
The previous CGM simulator generates ideal glucose curves. Now I need 
to add realistic CGM sensor characteristics to make it more authentic.

Enhance src/data/synthetic/cgm_simulator.py with a CGMNoiseModel class:

1. Sensor Noise (Gaussian):
   - CGMs have ~5-10% MARD (Mean Absolute Relative Difference)
   - Add Gaussian noise: sigma = 0.07 × true_glucose
   - But correlate noise: consecutive readings are correlated (AR(1) model)
   - noise_t = 0.8 × noise_{t-1} + 0.2 × new_noise (autocorrelated)

2. Calibration Drift:
   - CGM sensors drift over their 14-day life
   - Add a slow drift: drift_factor = 1 + 0.005 × day_in_sensor_life
   - Apply: measured = (true_glucose + noise) × drift_factor

3. CGM Lag:
   - CGMs measure interstitial glucose, not blood glucose
   - Interstitial glucose lags blood glucose by 7-15 minutes
   - Implement as: interstitial_t = glucose_t - lag_minutes × glucose_roc
   - lag_minutes ~ Uniform(7, 15) per patient

4. Missing Data Simulation:
   - CGMs have ~5% data loss (poor adhesion, signal loss)
   - Randomly mark 5% of readings as NaN
   - Longer gaps (15-30 min) during vigorous exercise (sensor displacement)

5. Compression Artifacts:
   - CGMs report integer mg/dL values (round to nearest integer)
   - Min reportable: 40 mg/dL, Max: 400 mg/dL (clip to range)

6. CGMNoiseModel.apply(clean_glucose_series) → noisy_series
   Returns the realistic CGM output given clean glucose input.
   Also return: data_quality_flag per reading (good/noisy/missing)
```

---

## 🤖 Vibe Prompt 2 — HRV from R-R Interval Simulation

```
Real HRV (Heart Rate Variability) is computed from R-R intervals 
(time between heartbeats). I want to simulate this realistically.

Create src/data/synthetic/hrv_simulator.py with HRVSimulator class:

1. RR Interval Generation:
   Generate a stream of R-R intervals (in milliseconds) at ~1Hz:
   - Base heart rate from patient profile: 
     * Resting HR = 60 + age/10 + (0 if active, +10 if sedentary)
     * Base RR = 60000 / HR_bpm (milliseconds)
   - HRV added as correlated noise following realistic spectral properties:
     * LF band (0.04-0.15 Hz): sympathetic + parasympathetic
     * HF band (0.15-0.40 Hz): parasympathetic (respiration-linked)
     * Use inverse FFT to generate correlated HRV signal

2. Compute Standard HRV Metrics from RR intervals (over 5-min windows):
   - RMSSD = sqrt(mean(diff(RR)^2)) — parasympathetic activity
   - SDNN = std(RR) — overall HRV
   - LF power, HF power (via Welch's periodogram)
   - LF/HF ratio — sympathovagal balance

3. Pathological Modulations (critical for prediction):
   When glucose < 80 mg/dL (hypoglycaemia approaching):
   - RMSSD drops to 50-70% of patient baseline (autonomic response)
   - LF/HF ratio increases (sympathetic activation)
   - HR increases (+10-20 bpm)
   
   When glucose > 180 mg/dL (hyperglycaemia):
   - RMSSD drops to 70-85% of baseline (vagal suppression)
   - Less dramatic than hypoglycaemia effect
   
   During exercise:
   - RMSSD drops (normal exercise response)
   - HR increases dramatically
   
   During deep sleep:
   - RMSSD highest of the day (parasympathetic dominance)
   - HR lowest (~50-60 bpm)

4. HRVSimulator.simulate(hours: int, patient_ehr: dict, 
                          glucose_series: pd.Series) → pd.DataFrame
   Returns 5-min averaged HRV metrics aligned with CGM timestamps.

Include unit tests verifying that:
- RMSSD drops during simulated hypoglycaemia
- RMSSD increases during sleep
- Values are within physiological ranges (RMSSD: 10-80ms for T2D patients)
```

---

## 🤖 Vibe Prompt 3 — Ohio T1DM Dataset Adapter

```
The Ohio T1DM dataset contains real CGM + wristband sensor data from 
12 T1D patients over 8 weeks. I want to adapt it for my T2D twin project.

Create src/data/ingestion/ohio_parser.py with OhioDatasetParser class:

1. Parse the Ohio XML format:
   Each patient has a {patient_id}-ws-training.xml with:
   - glucose_level entries: ts (timestamp), value (mg/dL)
   - finger_stick entries: ts, value
   - basal entries: ts, value (insulin units)
   - meal entries: ts, carbs (grams)
   - sleep entries: ts, duration
   - exercise entries: ts, intensity, duration
   - basis_heart_rate: ts, value
   - basis_gsr: ts, value (galvanic skin response)

2. Normalize to our schema:
   - Convert to 5-minute intervals (resample + interpolate)
   - Map: glucose_level → glucose_mgdl
   - Map: basis_heart_rate → heart_rate_bpm (compute HRV proxy)
   - Map: meal entries → meal_event flag + carbs_grams
   - Map: exercise → activity_type + activity_intensity

3. T1D → T2D adaptation:
   Ohio is T1D data. Adapt for T2D by:
   - Scaling glucose variability (T2D is less volatile than T1D)
     multiply (glucose - mean) by 0.7, add back mean
   - Adjusting event frequency (T2D has ~40% fewer severe events)
   - Note this adaptation in code comments + README

4. OhioDatasetParser.parse_patient(xml_path: Path) → pd.DataFrame
   OhioDatasetParser.parse_all(ohio_dir: Path) → Dict[str, pd.DataFrame]
   OhioDatasetParser.to_glucotwin_schema(df) → pd.DataFrame

5. If Ohio dataset is not available, raise OhioDataNotAvailableError 
   with a helpful message directing to the backup synthetic generator.

Include tests for the XML parsing logic.
```

---

## 🤖 Vibe Prompt 4 — Wearable Stream Manager (Real-time)

```
I need a WearableStreamManager that simulates real-time wearable sensor 
data delivery for the Digital Twin engine.

Create src/data/ingestion/wearable_stream.py:

class WearableStreamManager:
    """
    Manages real-time (or accelerated demo) wearable streams 
    for all active patient twins.
    """

    def __init__(self, demo_mode: bool = True, 
                 speed_multiplier: float = 60.0):
        """
        demo_mode: True = accelerated playback (5 min → 5 sec)
        speed_multiplier: 60.0 = 1 minute of data per second
        """

    async def start_patient_stream(self, patient_id: str, 
                                    historical_data: pd.DataFrame) → None:
        """Begin streaming pre-generated data for a patient."""

    async def get_next_reading(self, patient_id: str) → WearableReading:
        """Get the next sensor reading for a patient (async generator)."""

    async def inject_event(self, patient_id: str, 
                            event_type: str) → None:
        """
        For demo: inject specific events:
        - "post_meal_spike" → triggers hyperglycaemia trajectory
        - "insulin_overdose" → triggers hypoglycaemia trajectory  
        - "exercise_burst" → glucose dip + HRV suppression
        """

    def get_stream_status(self, patient_id: str) → dict:
        """Get current stream position, speed, patient count."""

    async def stop_patient_stream(self, patient_id: str) → None:
        """Stop streaming for a patient."""

Implement using asyncio and Python queues for clean async operation.
The stream should be injectable (for demo injection) and pauseable.

Also create a WearableReading dataclass:

@dataclass
class WearableReading:
    patient_id: str
    timestamp: datetime
    glucose_mgdl: float
    glucose_roc: float       # mg/dL/min
    glucose_acc: float       # mg/dL/min²
    heart_rate_bpm: float
    hrv_rmssd_ms: float
    hrv_sdnn_ms: float
    hrv_lf_hf_ratio: float
    steps_last_5min: int
    activity_type: str
    sleep_stage: str
    sleep_quality: float
    data_quality: str        # "good" / "noisy" / "missing"
    source: str              # "cgm_sim" / "ohio" / "real_device"
```

---

## 🤖 Vibe Prompt 5 — Wearable EDA Notebook

```
Create notebooks/02_wearable_eda.ipynb for exploratory analysis of 
the synthetic wearable dataset.

Section 1: CGM Data Overview
- Plot 5 sample patients' 24-hour glucose profiles on one chart
- Annotate: meal events, insulin events, exercise events
- Show glucose value distribution (histogram + kde)
- CGM data quality: % missing by patient

Section 2: Adverse Event Analysis
- Timeline plot showing all adverse events for one patient (30 days)
- Distribution of event types (hyper vs hypo)
- Event duration histogram
- Time-of-day heatmap of adverse events (shows dawn phenomenon)

Section 3: HRV-Glucose Correlation
- Scatter plot: RMSSD vs Glucose (colored by activity type)
- Time series: RMSSD and Glucose overlaid for 48 hours showing the 
  HRV suppression before hypoglycaemia event
- Correlation heatmap: RMSSD, SDNN, glucose, steps, sleep_quality

Section 4: Activity Patterns
- 24-hour average activity by hour (bar chart)
- Steps vs Glucose change (scatter, 1h lag)
- Sleep architecture for sample patient (hypnogram)

Section 5: Prediction Feature Preview
- For 10 random adverse events, show the preceding 4-hour window
  of: glucose, HRV, steps — all on same plot
- This visually demonstrates why the model can predict events early

Section 6: Class Balance Analysis
- Pie chart: label=1 (event) vs label=0 (no event)
- Event frequency by: hour of day, day of week, activity type
- Patient-level adverse event rate distribution

Use plotly for interactive charts where possible (better for demos).
```

---

## ✅ Validation Steps

```bash
# Test CGM noise model
python -c "
from src.data.synthetic.cgm_simulator import CGMNoiseModel
import numpy as np
clean = np.full(100, 150.0)  # Perfect 150 mg/dL signal
noisy = CGMNoiseModel().apply(clean)
print(f'Clean: {clean.mean():.1f}, Noisy: {noisy.mean():.1f}')
print(f'Noise std: {(noisy - clean).std():.2f} (expect ~10-15)')
"

# Test HRV simulator
python -c "
from src.data.synthetic.hrv_simulator import HRVSimulator
sim = HRVSimulator()
# Test that HRV drops during hypoglycaemia
import pandas as pd
import numpy as np
glucose_hypo = pd.Series(np.linspace(120, 65, 288))  # Glucose dropping to 65
hrv_df = sim.simulate(24, {'age': 50, 'bmi': 26, 'has_hypertension': True}, glucose_hypo)
print(f'HRV at normal glucose: {hrv_df[\"hrv_rmssd_ms\"].iloc[:50].mean():.1f}ms')
print(f'HRV at low glucose: {hrv_df[\"hrv_rmssd_ms\"].iloc[-50:].mean():.1f}ms')
# Second should be lower
"

# Test WearableStreamManager
python -c "
import asyncio
from src.data.ingestion.wearable_stream import WearableStreamManager
async def test():
    mgr = WearableStreamManager(demo_mode=True, speed_multiplier=1000)
    reading = await mgr.get_next_reading('TEST-001')
    print(f'Reading: glucose={reading.glucose_mgdl:.1f}, hrv={reading.hrv_rmssd_ms:.1f}ms')
asyncio.run(test())
"
```

---

## 📋 Session 04 Completion Checklist

- [ ] CGM noise model adds realistic sensor artifacts
- [ ] HRV simulator shows RMSSD drop during simulated hypoglycaemia
- [ ] Ohio parser handles XML format (or graceful fallback)
- [ ] WearableStreamManager delivers async readings per patient
- [ ] Event injection works (inject "post_meal_spike" triggers response)
- [ ] Demo mode at 60× speed (5 min of data in 5 seconds)
- [ ] Wearable EDA notebook complete (02_wearable_eda.ipynb)
- [ ] All unit tests pass

## 🔜 Next Session

→ **05-digital-twin-engine.md** — The PatientTwin class and state management

---

*GlucoTwin Vibe Assist 04 | Wearable Integration | Digital Twin Challenge 2026*
