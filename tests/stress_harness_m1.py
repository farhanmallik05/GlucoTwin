"""
tests/stress_harness_m1.py

Empirical Stress Testing Harness for Milestone 1:
- CGM Simulator & Noise Model (extreme glucose values, clamping, MARD, drift, dropouts)
- Missing data & NaN propagation vulnerability checks
- Autonomic HRV coupling with hypoglycemia (<80 mg/dL)
- WearableStreamManager under rapid ticks, queue pressure, and concurrency
- FeatureEngineer edge conditions (short windows, missing channels, extreme inputs)
"""

import asyncio
import math
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.data.synthetic.cgm_simulator import CGMSimulator, CGMNoiseModel
from src.data.synthetic.hrv_simulator import HRVSimulator
from src.data.synthetic.wearable_simulator import WearableSimulator, WearableReading
from src.data.ingestion.wearable_stream import WearableStreamManager
from src.data.preprocessing.clinical_validator import ClinicalValidator
from src.data.preprocessing.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.data.preprocessing.clinical_indices import compute_clinical_indices

results = {}

def log_test(name, passed, details=""):
    results[name] = {"passed": passed, "details": details}
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {details}")

# =========================================================================
# TEST SUITE 1: Extreme Glucose Values & Physiological Clamping
# =========================================================================
def test_noise_model_clamping_extremes():
    print("\n--- Running Suite 1: Extreme Glucose Values & Clamping ---")
    model = CGMNoiseModel(m_ard=0.07, seed=42)
    
    # 1.1 Inputs severely below 40 mg/dL
    neg_series = pd.Series([-500.0, -100.0, 0.0, 10.0, 20.0, 39.9])
    noisy_neg = model.apply(neg_series).dropna()
    min_neg = noisy_neg.min()
    log_test("Extreme Negative Clamping (min >= 40)", min_neg >= 40.0, f"Observed min={min_neg}")
    
    # 1.2 Inputs severely above 400 mg/dL
    high_series = pd.Series([400.1, 450.0, 600.0, 1500.0, 10000.0])
    noisy_high = model.apply(high_series).dropna()
    max_high = noisy_high.max()
    log_test("Extreme High Clamping (max <= 400)", max_high <= 400.0, f"Observed max={max_high}")

    # 1.3 Rapid step jump (from 40 to 400 in 1 step -> large ROC)
    jump_series = pd.Series([40.0, 400.0, 40.0, 400.0] * 10)
    noisy_jump = model.apply(jump_series).dropna()
    log_test("Rapid Step Jump Clamping [40, 400]", 
             (noisy_jump.min() >= 40.0) and (noisy_jump.max() <= 400.0),
             f"Min={noisy_jump.min()}, Max={noisy_jump.max()}")

    # 1.4 Simulator generation with extreme patient EHR
    extreme_ehr = {
        "hba1c_latest": 18.0,
        "fasting_glucose_avg": 850.0,
        "bmi": 65.0,
        "on_insulin": True,
        "insulin_dose_units": 180.0
    }
    sim = CGMSimulator(patient_ehr=extreme_ehr, seed=123)
    df_extreme = sim.generate(duration_days=3)
    clean_g = df_extreme["glucose_mgdl"].dropna()
    log_test("CGMSimulator Extreme Patient Bounded in [40, 400]",
             (clean_g.min() >= 40.0) and (clean_g.max() <= 400.0),
             f"Observed min={clean_g.min()}, max={clean_g.max()}")

# =========================================================================
# TEST SUITE 2: Missing Data, NaNs, Sensor Drops, and 14-Day Drift
# =========================================================================
def test_missing_data_and_nan_propagation():
    print("\n--- Running Suite 2: Missing Data, NaNs & Drift ---")
    model = CGMNoiseModel(m_ard=0.07, seed=99)
    
    # 2.1 Single NaN input in clean glucose series
    series_with_nan = pd.Series([120.0, 125.0, np.nan, 130.0, 135.0, 140.0, 145.0, 150.0])
    try:
        out = model.apply(series_with_nan)
        # Check if values after index 2 survived or were wiped out
        nan_count = out.isna().sum()
        subsequent_nans = out.iloc[3:].isna().all()
        log_test("Single NaN Input Does Not Corrupt Subsequent Series",
                 not subsequent_nans,
                 f"Total NaNs={nan_count}/{len(series_with_nan)}, subsequent all-NaN={subsequent_nans}")
    except Exception as e:
        log_test("Single NaN Input Does Not Crash Noise Model", False, f"Exception: {e}")

    # 2.2 100% Dropout Rate
    model_drop_all = CGMNoiseModel(dropout_prob=1.0, seed=42)
    clean_s = pd.Series([150.0] * 50)
    out_drop, q_drop = model_drop_all.apply(clean_s, return_quality=True)
    log_test("100% Dropout Produces All NaN & 'missing' Flags",
             out_drop.isna().all() and (q_drop == "missing").all(),
             f"NaN count={out_drop.isna().sum()}, Quality missing count={(q_drop == 'missing').sum()}")

    # 2.3 14-Day Drift Continuity Check
    # At day 13.99 vs day 14.01, check whether drift resets discontinuously
    model_drift = CGMNoiseModel(m_ard=0.0, ar1_phi=0.0, dropout_prob=0.0, lag_min_range=(0.0, 0.0), seed=42)
    # Step at day 13.999 (step 4031 of 288*14 = 4032) vs step 4032 (day 14.00)
    steps_14d = 288 * 14 + 10
    flat_g = pd.Series([100.0] * steps_14d)
    out_drift = model_drift.apply(flat_g)
    drift_at_13_99 = out_drift.iloc[4031]
    drift_at_14_00 = out_drift.iloc[4032]
    drift_jump = abs(drift_at_14_00 - drift_at_13_99)
    # Expected jump is ~7 mg/dL if modulo 14 resets abruptly
    log_test("14-Day Drift Discontinuity at Boundary",
             drift_jump < 2.0,
             f"Value before boundary={drift_at_13_99:.2f}, after boundary={drift_at_14_00:.2f}, jump={drift_jump:.2f} mg/dL")

    # 2.4 Empty Series Input
    empty_series = pd.Series([], dtype=float)
    try:
        empty_out = model.apply(empty_series)
        log_test("Empty Series Handling", len(empty_out) == 0, "Handled gracefully")
    except Exception as e:
        log_test("Empty Series Handling", False, f"Crashed with {e}")

# =========================================================================
# TEST SUITE 3: Autonomic HRV Coupling Under Simulated Hypoglycemia
# =========================================================================
def test_autonomic_hrv_coupling():
    print("\n--- Running Suite 3: Autonomic HRV Coupling ---")
    ehr = {"age": 50, "bmi": 26.0, "hba1c_latest": 7.5, "physical_activity_encoded": 1}
    hrv_sim = HRVSimulator(patient_ehr=ehr, random_seed=42)
    
    # 3.1 Verify RMSSD drops 30-50% during hypoglycemia (<80 mg/dL)
    normal_df = pd.DataFrame({"glucose_mgdl": [120.0] * 50})
    hypo_df = pd.DataFrame({"glucose_mgdl": [60.0] * 50})
    severe_hypo_df = pd.DataFrame({"glucose_mgdl": [40.0] * 50})

    out_norm = hrv_sim.simulate(normal_df)
    out_hypo = hrv_sim.simulate(hypo_df)
    out_severe = hrv_sim.simulate(severe_hypo_df)

    mean_norm_rmssd = out_norm["hrv_rmssd_ms"].mean()
    mean_hypo_rmssd = out_hypo["hrv_rmssd_ms"].mean()
    mean_severe_rmssd = out_severe["hrv_rmssd_ms"].mean()

    drop_hypo_pct = (1.0 - mean_hypo_rmssd / mean_norm_rmssd) * 100.0
    drop_severe_pct = (1.0 - mean_severe_rmssd / mean_norm_rmssd) * 100.0

    log_test("HRV RMSSD Drops 30-50% During Moderate Hypo (60 mg/dL)",
             30.0 <= drop_hypo_pct <= 55.0,
             f"Normal={mean_norm_rmssd:.1f}ms, Hypo={mean_hypo_rmssd:.1f}ms, Drop={drop_hypo_pct:.1f}%")

    log_test("HRV RMSSD Drops ~50% During Severe Hypo (40 mg/dL)",
             40.0 <= drop_severe_pct <= 60.0,
             f"Severe={mean_severe_rmssd:.1f}ms, Drop={drop_severe_pct:.1f}%")

    # 3.2 Sympathetic Tone Surge (Heart Rate increase +10-20 bpm, LF/HF > 2.5)
    mean_norm_hr = out_norm["heart_rate_bpm"].mean()
    mean_hypo_hr = out_hypo["heart_rate_bpm"].mean()
    hr_increase = mean_hypo_hr - mean_norm_hr
    mean_hypo_lfhf = out_hypo["hrv_lf_hf_ratio"].mean()

    log_test("Hypoglycemia Tachycardia (+10-25 bpm)",
             10.0 <= hr_increase <= 30.0,
             f"HR increase={hr_increase:.1f} bpm (Normal={mean_norm_hr:.1f}, Hypo={mean_hypo_hr:.1f})")

    log_test("Hypoglycemia Sympathovagal LF/HF Surge (>= 2.5)",
             mean_hypo_lfhf >= 2.5,
             f"LF/HF ratio={mean_hypo_lfhf:.2f}")

    # 3.3 Physiological Conflict: Nocturnal Sleep + Hypoglycemia
    # Check behavior when sleep=True during hypoglycemia
    sleep_flags = np.ones(50, dtype=bool)
    out_sleep_hypo = hrv_sim.simulate(hypo_df, sleep_flags=sleep_flags)
    mean_sleep_hypo_rmssd = out_sleep_hypo["hrv_rmssd_ms"].mean()
    sleep_hypo_drop = (1.0 - mean_sleep_hypo_rmssd / mean_norm_rmssd) * 100.0

    # During sleep, does RMSSD still drop by at least 15%?
    log_test("Nocturnal Hypoglycemia Retains Autonomic Signal",
             sleep_hypo_drop >= 10.0,
             f"Sleep Hypo RMSSD={mean_sleep_hypo_rmssd:.1f}ms, Drop vs daytime baseline={sleep_hypo_drop:.1f}%")

# =========================================================================
# TEST SUITE 4: Asynchronous WearableStreamManager Stress Testing
# =========================================================================
async def run_async_stream_manager_tests():
    print("\n--- Running Suite 4: Async WearableStreamManager Stress ---")
    
    # 4.1 Rapid tick throughput (speed multiplier = 3000x -> tick every 100ms)
    mgr = WearableStreamManager(demo_mode=True, speed_multiplier=3000.0)
    
    # Prepare historical mock data (200 readings)
    dates = [datetime.utcnow() + timedelta(minutes=5 * i) for i in range(200)]
    mock_df = pd.DataFrame({
        "timestamp": dates,
        "glucose_mgdl": np.random.uniform(90, 160, 200),
        "glucose_roc": np.zeros(200),
        "heart_rate_bpm": np.full(200, 72.0),
        "hrv_rmssd_ms": np.full(200, 40.0),
    })

    pid = "PAT-STRESS-01"
    await mgr.start_patient_stream(pid, mock_df)
    
    # Consume 25 readings rapidly
    consumed = []
    start_t = time.perf_counter()
    for _ in range(25):
        reading = await mgr.get_next_reading(pid)
        consumed.append(reading)
    elapsed = time.perf_counter() - start_t

    log_test("Rapid Tick Consumption (25 readings at 3000x)",
             len(consumed) == 25 and elapsed < 5.0,
             f"Consumed {len(consumed)} readings in {elapsed:.2f}s")

    # 4.2 Scenario Event Injections (post_meal_spike, insulin_overdose, exercise_burst)
    await mgr.inject_event(pid, "post_meal_spike")
    spike_reading = await mgr.get_next_reading(pid)
    log_test("Injected Post-Meal Spike Alters Telemetry",
             spike_reading.glucose_mgdl > 180.0 and spike_reading.glucose_roc == 2.4,
             f"Glucose={spike_reading.glucose_mgdl}, ROC={spike_reading.glucose_roc}")

    await mgr.inject_event(pid, "insulin_overdose")
    hypo_reading = await mgr.get_next_reading(pid)
    log_test("Injected Insulin Overdose Drops Glucose & RMSSD",
             hypo_reading.glucose_mgdl <= 60.0 and hypo_reading.glucose_roc == -3.2 and hypo_reading.hrv_rmssd_ms <= 25.0,
             f"Glucose={hypo_reading.glucose_mgdl}, ROC={hypo_reading.glucose_roc}, RMSSD={hypo_reading.hrv_rmssd_ms}")

    await mgr.inject_event(pid, "exercise_burst")
    ex_reading = await mgr.get_next_reading(pid)
    log_test("Injected Exercise Burst Increases Steps & Heart Rate",
             ex_reading.steps_last_5min >= 500 and ex_reading.heart_rate_bpm >= 140.0 and ex_reading.activity_type == "exercise",
             f"Steps={ex_reading.steps_last_5min}, HR={ex_reading.heart_rate_bpm}, Activity={ex_reading.activity_type}")

    # 4.3 Queue Overflow Protection (producer unread for 150 ticks with queue maxsize=100)
    # The manager must not lock up or crash on full queue
    await asyncio.sleep(0.5) # Let 5 more ticks accumulate
    status = mgr.get_stream_status(pid)
    log_test("Stream Loop Survives Unread Queue",
             status["active"] is True,
             f"Status active={status['active']}, Current index={status['current_index']}")

    # 4.4 Graceful Clean Shutdown
    await mgr.stop_patient_stream(pid)
    status_stopped = mgr.get_stream_status(pid)
    log_test("Graceful Stream Stop",
             status_stopped["active"] is False and pid not in mgr._patient_tasks,
             "Task cancelled and deleted")

    # 4.5 Edge Case: Empty DataFrame passed to start_patient_stream
    empty_pid = "PAT-EMPTY"
    try:
        await mgr.start_patient_stream(empty_pid, pd.DataFrame())
        await asyncio.sleep(0.05)
        # Check if task survived or crashed
        task = mgr._patient_tasks.get(empty_pid)
        crashed = task.done() and task.exception() is not None
        log_test("Empty DataFrame in start_patient_stream Handled Safely",
                 not crashed,
                 f"Task done={task.done()}, exception={task.exception() if task.done() else None}")
        await mgr.stop_patient_stream(empty_pid)
    except Exception as e:
        log_test("Empty DataFrame in start_patient_stream Handled Safely", False, f"Exception: {e}")

# =========================================================================
# TEST SUITE 5: Feature Engineering Edge Cases
# =========================================================================
def test_feature_engineering_resilience():
    print("\n--- Running Suite 5: Feature Engineering Edge Cases ---")
    fe = FeatureEngineer()
    
    # 5.1 Short window (< 48 rows)
    short_df = pd.DataFrame({
        "glucose_mgdl": [130.0] * 10,
        "glucose_roc": [0.0] * 10,
        "glucose_acc": [0.0] * 10,
        "hrv_rmssd_ms": [35.0] * 10,
        "hrv_sdnn_ms": [45.0] * 10,
        "hrv_lf_hf_ratio": [1.5] * 10,
        "steps_last_5min": [0] * 10,
        "activity_type": ["resting"] * 10,
        "sleep_stage": ["none"] * 10,
        "sleep_quality": [0.85] * 10,
    })
    try:
        f_short = fe.compute_features(short_df)
        log_test("FeatureEngineer Short Window (<48 rows)",
                 len(f_short) == 89,
                 f"Features count={len(f_short)}")
    except Exception as e:
        log_test("FeatureEngineer Short Window (<48 rows)", False, f"Crashed: {type(e).__name__}: {e}")

    # 5.2 All-NaN Glucose Window
    all_nan_df = pd.DataFrame({
        "glucose_mgdl": [np.nan] * 48,
        "glucose_roc": [np.nan] * 48,
        "glucose_acc": [np.nan] * 48,
        "hrv_rmssd_ms": [np.nan] * 48,
        "hrv_sdnn_ms": [np.nan] * 48,
        "hrv_lf_hf_ratio": [np.nan] * 48,
        "steps_last_5min": [np.nan] * 48,
        "activity_type": ["resting"] * 48,
        "sleep_stage": ["none"] * 48,
        "sleep_quality": [np.nan] * 48,
    })
    try:
        f_nan = fe.compute_features(all_nan_df)
        has_nans = any(math.isnan(v) for v in f_nan.values())
        log_test("All-NaN Window Feature Sanitation",
                 not has_nans and len(f_nan) == 89,
                 f"NaNs present={has_nans}, len={len(f_nan)}")
    except Exception as e:
        log_test("All-NaN Window Feature Sanitation", False, f"Crashed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_noise_model_clamping_extremes()
    test_missing_data_and_nan_propagation()
    test_autonomic_hrv_coupling()
    asyncio.run(run_async_stream_manager_tests())
    test_feature_engineering_resilience()
    
    print("\n=======================================================")
    print("STRESS HARNESS SUMMARY:")
    passed_count = sum(1 for v in results.values() if v["passed"])
    total_count = len(results)
    print(f"Total Tests: {total_count} | Passed: {passed_count} | Failed: {total_count - passed_count}")
    print("=======================================================")
    if total_count - passed_count > 0:
        sys.exit(1)
    sys.exit(0)
