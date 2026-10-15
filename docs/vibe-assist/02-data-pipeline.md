# Vibe Assist 02 — Data Pipeline
## EHR Ingestion + Wearable Simulation

> **Vibe Coding Session 02/10** | Estimated time: 4-5 hours
> Goal: A complete data pipeline from raw Synthea output to a unified training dataset.

---

## 🎯 Session Goal

By the end of this session you will have:
- ✅ `synthea_parser.py` — Parses Synthea FHIR JSON → flat patient DataFrame
- ✅ `cgm_simulator.py` — Generates realistic CGM time series for T2D patients
- ✅ `wearable_simulator.py` — Generates HRV, steps, sleep data
- ✅ `feature_engineering.py` — Computes all 89 features
- ✅ `labeler.py` — Labels adverse events (120-minute horizon)
- ✅ Unified `training_dataset.csv` ready for model training

---

## 🤖 Vibe Prompt 1 — Synthea FHIR Parser

```
I have Synthea-generated FHIR R4 JSON files for 1000 synthetic Type 2 
Diabetes patients. The files are in data/raw/synthea/{patient_id}/ 
directories, each containing: Patient.json, Condition.json, 
Observation.json, MedicationRequest.json, Encounter.json.

Create src/data/ingestion/synthea_parser.py that:

1. Defines a SyntheaParser class with methods:
   - parse_patient(patient_dir: Path) → dict
   - parse_all(synthea_dir: Path) → pd.DataFrame

2. Extracts these features from FHIR JSON:
   Patient.json:    age, gender, ethnicity, city, state
   Condition.json:  diabetes_duration_years, has_hypertension, 
                    has_dyslipidaemia, has_neuropathy, has_retinopathy
   Observation.json: hba1c_latest, hba1c_6m_ago (trend), 
                     fasting_glucose_avg, egfr, blood_pressure_sys, bmi
   MedicationRequest.json: on_metformin, on_insulin, insulin_dose_units,
                           on_sulphonylurea, on_sglt2

3. Returns a DataFrame where each row = one patient, columns = features

4. Handles missing values gracefully:
   - Numeric: fill with population median
   - Boolean: fill with False (assume no comorbidity if not recorded)
   - Flag any patient with > 30% missing features as LOW_QUALITY

5. Includes a __main__ block that:
   - Runs the parser on data/raw/synthea/
   - Saves output to data/processed/ehr_features.csv
   - Prints: total patients, feature completeness stats

6. Full type hints and docstrings throughout.
7. Use pandas 2.0 API.
```

---

## 🤖 Vibe Prompt 2 — CGM Simulator

```
I need a CGM (Continuous Glucose Monitor) time-series simulator for 
Type 2 Diabetes patients that generates realistic glucose patterns.

Create src/data/synthetic/cgm_simulator.py with:

1. A CGMSimulator class that takes a patient EHR profile (dict) and 
   generates CGM readings every 5 minutes over N days.

2. The simulation must capture real T2D glucose patterns:
   - Fasting baseline based on hba1c (e.g., HbA1c 7.5% → ~170 mg/dL fasting avg)
   - Post-meal spikes (breakfast: +60-120 mg/dL, peaks at 45-90 min)
   - Dawn phenomenon (glucose rises 4-8 AM naturally)
   - Exercise effect (glucose drops ~30 mg/dL during moderate activity)
   - Insulin correction effect (if on_insulin: pharmacokinetic decay)
   - Random glucose variability (patient-specific sigma based on HbA1c)

3. Must simulate ADVERSE EVENTS at a ~15% hourly probability when:
   - Fasting glucose > 200 mg/dL at baseline (hyperglycaemic patient)
   - Insulin was taken but meal was delayed (hypoglycaemia dip to 55-70)
   - High-carb meal without correction (spike to 260-320)

4. Output format:
   DataFrame with columns: patient_id, timestamp (5-min intervals), 
   glucose_mgdl, glucose_roc (rate of change, mg/dL/min), 
   glucose_acc (acceleration, 2nd derivative),
   meal_event (bool), exercise_event (bool), insulin_event (bool)

5. Reproducible with random seed.

6. __main__ block: generate for 10 demo patients over 30 days, 
   save to data/synthetic/cgm_demo.csv, plot 3 sample patients.
```

---

## 🤖 Vibe Prompt 3 — Wearable Simulator (HRV + Steps + Sleep)

```
I need a wearable sensor simulator for Type 2 Diabetes patients that 
generates HRV, step count, and sleep stage data, correlated with the 
CGM glucose state.

Create src/data/synthetic/wearable_simulator.py with a WearableSimulator 
class that takes:
  - patient_ehr: dict (patient profile from EHR)
  - cgm_df: pd.DataFrame (from CGMSimulator, already generated)

And adds these columns to the CGM DataFrame (same timestamps):

HRV features (every 5-min):
  hrv_rmssd_ms      → Normal: 20-50ms. Drops to 10-25ms during 
                       hypoglycaemia (key predictor!). Lower in older/obese.
  hrv_sdnn_ms       → Normal: 30-80ms. Correlated with RMSSD.
  hrv_lf_hf_ratio   → LF/HF balance, inversely related to RMSSD.

Activity features (every 5-min):
  steps_last_5min   → 0 during sleep/rest, 50-150 during walking,
                       200-600 during exercise.
  steps_last_hour   → Rolling sum of steps_last_5min × 12 windows.
  activity_type     → Categorical: "resting", "walking", "exercise", "sleep"
  calories_5min     → Estimated from steps + patient weight (from BMI + height)

Sleep features (during night hours):
  sleep_stage       → "awake", "light", "deep", "rem" 
                       (cycles through realistic 90-min sleep architecture)
  sleep_quality     → 0-1 score (lower in poorly-controlled diabetics)

Clinical correlation rules:
  - HRV drops 30-50% during hypoglycaemia events (glucose < 80 mg/dL)
  - HRV drops 15-25% during hyperglycaemia (glucose > 200 mg/dL)
  - Exercise → immediate glucose drop in cgm (feed back to CGM sim)
  - Poor sleep → elevated morning glucose (+20-40 mg/dL dawn phenomenon)

Output: Combined DataFrame with all CGM + wearable columns.
Include docstrings, type hints, and a __main__ demo.
```

---

## 🤖 Vibe Prompt 4 — Feature Engineering Pipeline

```
I have a combined DataFrame with CGM + wearable columns for T2D patients.
Create src/data/preprocessing/feature_engineering.py that computes 
89 features for each timestep (for ML model input).

The FeatureEngineer class should have a method:
  compute_features(window_df: pd.DataFrame, ehr_features: dict) → dict

Where window_df is the last 4 hours (48 rows at 5-min intervals) of:
  [glucose_mgdl, glucose_roc, glucose_acc, hrv_rmssd_ms, hrv_sdnn_ms,
   steps_last_5min, activity_type, sleep_stage, sleep_quality]

And ehr_features is the static patient profile dict.

Compute these 89 features:

EHR Static Features (23):
  All columns from ehr_features dict as-is
  (age, gender_encoded, bmi, hba1c_latest, hba1c_trend, fasting_glucose_avg,
   egfr, bp_sys, diabetes_duration, on_metformin, on_insulin, 
   insulin_dose_units, on_sulphonylurea, on_sglt2, has_hypertension,
   has_dyslipidaemia, has_neuropathy, has_retinopathy, bmi_category_encoded,
   age_group_encoded, ethnicity_encoded, physical_activity_encoded,
   dietary_compliance)

CGM Statistics - multiple windows (20):
  glucose_mean_1h, glucose_std_1h, glucose_min_1h, glucose_max_1h
  glucose_mean_4h, glucose_std_4h, glucose_min_4h, glucose_max_4h
  glucose_mean_12h, glucose_std_12h, glucose_min_12h, glucose_max_12h  
  glucose_mean_24h, glucose_std_24h
  glucose_roc_mean_30min, glucose_roc_std_30min
  glucose_acc_mean_15min
  glucose_percentile_10_4h, glucose_percentile_90_4h, glucose_iqr_4h

CGM Derived Features (8):
  glucose_current, glucose_roc_current, glucose_acc_current
  glucose_excursion_count_4h  (times crossed 180 or 70 threshold)
  glucose_above_180_pct_4h   (% time above 180)
  glucose_below_70_pct_4h    (% time below 70)
  glucose_coefficient_of_variation_4h
  glucose_personal_deviation  (current - patient_baseline_mean)

HRV Features (12):
  hrv_rmssd_mean_30min, hrv_rmssd_std_30min, hrv_rmssd_min_30min
  hrv_rmssd_mean_4h, hrv_rmssd_trend_4h  (slope of RMSSD)
  hrv_sdnn_mean_1h, hrv_sdnn_std_1h
  hrv_lf_hf_mean_1h
  hrv_suppression_flag  (1 if RMSSD dropped > 40% in last 30 min)
  hrv_personal_deviation  (vs patient baseline)
  hrv_rmssd_pct_change_30min
  hrv_recovery_score  (RMSSD / max(RMSSD_last_24h))

Activity Features (8):
  steps_last_hour, steps_last_4h, steps_last_24h
  activity_type_encoded  (resting=0, walking=1, exercise=2, sleep=-1)
  is_exercising  (bool)
  is_sleeping    (bool)
  time_since_exercise_min  (minutes since last exercise session)
  calories_last_hour

Sleep Features (6):
  sleep_stage_encoded  (awake=0, light=1, deep=2, rem=3)
  sleep_quality_current
  sleep_total_hours_last_night
  sleep_efficiency_last_night
  sleep_deep_pct_last_night
  is_night_hours  (bool: 22:00-06:00)

Temporal Features (6):
  hour_of_day  (0-23)
  day_of_week  (0-6)
  is_weekend   (bool)
  time_since_last_meal_min  (estimated from glucose spike pattern)
  time_since_last_insulin_min  (from insulin event log)
  circadian_phase  (0-1, normalized hour)

Clinical Indices (6):
  tir_24h   (% time in range 70-180 over last 24h)
  tar_24h   (% time above range > 180)
  tbr_24h   (% time below range < 70)
  gmi       (Glucose Management Indicator: 3.31 + 0.02392 × mean_glucose)
  cv_glucose_24h  (coefficient of variation)
  io_board  (estimated insulin-on-board from last dose and time)

Return a flat dict of all 89 features with float/int values.
Include full docstrings, type hints, and unit tests in the same file.
```

---

## 🤖 Vibe Prompt 5 — Event Labeller + Dataset Builder

```
Create src/data/preprocessing/labeler.py and a dataset builder script.

1. In labeler.py:
   Create label_adverse_events(cgm_df: pd.DataFrame, 
                                horizon_minutes: int = 120) → pd.DataFrame

   For each row in cgm_df, label = 1 if glucose goes below 70 or 
   above 250 mg/dL at any point in the next horizon_minutes.
   
   Also add:
   - minutes_to_event: how many minutes until the event (NaN if no event)
   - event_type: "hypoglycaemia" | "hyperglycaemia" | "none"
   
   Class balance report: print % of rows with label=1.

2. Create scripts/build_dataset.py that:
   - Loads ehr_features.csv
   - For each patient, loads their cgm+wearable data
   - For each timestep (every 5 minutes), computes 89 features
   - Labels each timestep with adverse event label
   - Concatenates all patients into one DataFrame
   - Splits into train/val/test (80/10/10) by PATIENT (not by row)
   - Saves:
     data/processed/train.csv
     data/processed/val.csv
     data/processed/test.csv
     data/processed/dataset_stats.json (class balance, feature stats)
   
   Patient-level split is CRITICAL — prevents data leakage where the 
   model sees future data from the same patient during training.
   
   Print final stats: total rows, % adverse events, feature count.
```

---

## ✅ Manual Validation Steps

After running all vibe prompts, verify:

```bash
# 1. Check EHR parser output
python src/data/ingestion/synthea_parser.py
# Expected: "Parsed 1000 patients, 23 features, completeness: >95%"

# 2. Check CGM simulator
python src/data/synthetic/cgm_simulator.py
# Expected: plots showing realistic glucose curves with spikes

# 3. Check wearable simulator
python src/data/synthetic/wearable_simulator.py
# Expected: HRV drops visible during glucose lows

# 4. Check feature engineering
python -c "from src.data.preprocessing.feature_engineering import FeatureEngineer; print('OK')"

# 5. Build full dataset
python scripts/build_dataset.py
# Expected:
# Total rows: ~2.5 million (1000 patients × 30 days × 288 readings/day)
# Adverse event rate: ~12-18%
# Features: 89
# Train: 800 patients, Val: 100, Test: 100
```

---

## 📋 Session 02 Completion Checklist

- [ ] `synthea_parser.py` runs on Synthea output
- [ ] `ehr_features.csv` created (1000 patients × 23 features)
- [ ] `cgm_simulator.py` generates realistic glucose curves
- [ ] `wearable_simulator.py` adds HRV/steps/sleep with clinical correlations
- [ ] `feature_engineering.py` outputs exactly 89 features per timestep
- [ ] `labeler.py` correctly labels 120-min horizon events
- [ ] `build_dataset.py` creates train/val/test splits by patient
- [ ] Dataset class balance ~15% adverse events
- [ ] No data leakage (patient-level split confirmed)

## 🔜 Next Session

→ **03-ehr-simulation.md** — Deep dive into EHR data quality and augmentation

---

*GlucoTwin Vibe Assist 02 | Data Pipeline | Digital Twin Challenge 2026*
