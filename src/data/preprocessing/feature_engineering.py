"""
src/data/preprocessing/feature_engineering.py

Calculates the authoritative 89-feature vector for the GlucoTwin predictive models
(XGBoost tabular model and conditional SHAP TreeExplainer).
Combines:
- 23 Static EHR features
- 20 CGM multi-window statistical features (1h, 4h, 12h, 24h)
- 8 CGM derived features
- 12 HRV autonomic features
- 8 Activity features
- 6 Sleep features
- 6 Temporal features
- 6 Clinical indices & PK features (TIR, TAR, TBR, GMI, CV, IOB)
"""

from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.data.preprocessing.clinical_indices import compute_clinical_indices

# Authoritative 89-feature column list matching project specification exactly
FEATURE_COLUMNS: List[str] = [
    # 1. EHR Static Features (23)
    "age",
    "gender_encoded",
    "bmi",
    "hba1c_latest",
    "hba1c_trend",
    "fasting_glucose_avg",
    "egfr",
    "bp_sys",
    "diabetes_duration_years",
    "on_metformin",
    "on_insulin",
    "insulin_dose_units",
    "on_sulphonylurea",
    "on_sglt2",
    "has_hypertension",
    "has_dyslipidaemia",
    "has_neuropathy",
    "has_retinopathy",
    "bmi_category_encoded",
    "age_group_encoded",
    "ethnicity_encoded",
    "physical_activity_encoded",
    "dietary_compliance_score",

    # 2. CGM Multi-Window Statistics (20)
    "glucose_mean_1h",
    "glucose_std_1h",
    "glucose_min_1h",
    "glucose_max_1h",
    "glucose_mean_4h",
    "glucose_std_4h",
    "glucose_min_4h",
    "glucose_max_4h",
    "glucose_mean_12h",
    "glucose_std_12h",
    "glucose_min_12h",
    "glucose_max_12h",
    "glucose_mean_24h",
    "glucose_std_24h",
    "glucose_roc_mean_30min",
    "glucose_roc_std_30min",
    "glucose_acc_mean_15min",
    "glucose_percentile_10_4h",
    "glucose_percentile_90_4h",
    "glucose_iqr_4h",

    # 3. CGM Derived Features (8)
    "glucose_current",
    "glucose_roc_current",
    "glucose_acc_current",
    "glucose_excursion_count_4h",
    "glucose_above_180_pct_4h",
    "glucose_below_70_pct_4h",
    "glucose_coefficient_of_variation_4h",
    "glucose_personal_deviation",

    # 4. HRV Features (12)
    "hrv_rmssd_mean_30min",
    "hrv_rmssd_std_30min",
    "hrv_rmssd_min_30min",
    "hrv_rmssd_mean_4h",
    "hrv_rmssd_trend_4h",
    "hrv_sdnn_mean_1h",
    "hrv_sdnn_std_1h",
    "hrv_lf_hf_mean_1h",
    "hrv_suppression_flag",
    "hrv_personal_deviation",
    "hrv_rmssd_pct_change_30min",
    "hrv_recovery_score",

    # 5. Activity Features (8)
    "steps_last_hour",
    "steps_last_4h",
    "steps_last_24h",
    "activity_type_encoded",
    "is_exercising",
    "is_sleeping",
    "time_since_exercise_min",
    "calories_last_hour",

    # 6. Sleep Features (6)
    "sleep_stage_encoded",
    "sleep_quality_current",
    "sleep_total_hours_last_night",
    "sleep_efficiency_last_night",
    "sleep_deep_pct_last_night",
    "is_night_hours",

    # 7. Temporal Features (6)
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "time_since_last_meal_min",
    "time_since_last_insulin_min",
    "circadian_phase",

    # 8. Clinical Indices & PK (6)
    "tir_24h",
    "tar_24h",
    "tbr_24h",
    "gmi",
    "cv_glucose_24h",
    "io_board",
]

assert len(FEATURE_COLUMNS) == 89, f"Expected 89 feature columns, got {len(FEATURE_COLUMNS)}"


class FeatureEngineer:
    """
    Computes 89 engineered features from multi-modal wearable window and EHR profile.
    """

    def __init__(self, default_ehr: Optional[Dict[str, Any]] = None):
        self.default_ehr = default_ehr or {}

    def compute_features(
        self,
        window_df: pd.DataFrame,
        ehr_features: Optional[Dict[str, Any]] = None,
        iob: float = 0.0,
        baseline_glucose: Optional[float] = None,
        baseline_hrv: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Computes all 89 features for the latest timestep in window_df.

        Args:
            window_df: Sliding window DataFrame (up to 48 timesteps, 5-min intervals).
            ehr_features: Dictionary of 23 static EHR features.
            iob: Active insulin units (Insulin-On-Board).
            baseline_glucose: Personalized fasting/eAG glucose baseline in mg/dL.
            baseline_hrv: Personalized baseline RMSSD in ms.

        Returns:
            Dictionary with exactly 89 keys matching FEATURE_COLUMNS, all float values.
        """
        ehr = dict(self.default_ehr)
        if ehr_features:
            ehr.update(ehr_features)

        # Ensure window is valid and copy
        if window_df.empty:
            w_df = pd.DataFrame({
                "glucose_mgdl": [140.0] * 48,
                "glucose_roc": [0.0] * 48,
                "glucose_acc": [0.0] * 48,
                "hrv_rmssd_ms": [38.0] * 48,
                "hrv_sdnn_ms": [45.0] * 48,
                "hrv_lf_hf_ratio": [1.5] * 48,
                "steps_last_5min": [0] * 48,
                "activity_type": ["resting"] * 48,
                "sleep_stage": ["none"] * 48,
                "sleep_quality": [0.85] * 48,
            })
        else:
            w_df = window_df.copy()

        # Handle short window (< 48 rows) via left padding
        n_rows = len(w_df)
        if n_rows < 48:
            pad_len = 48 - n_rows
            first_row = w_df.iloc[[0]]
            pad_df = pd.concat([first_row] * pad_len, ignore_index=True)
            w_df = pd.concat([pad_df, w_df], ignore_index=True)

        # -------------------------------------------------------------
        # Extract and Impute Wearable Series
        # -------------------------------------------------------------
        # Glucose
        if "glucose_mgdl" in w_df.columns:
            g_series = w_df["glucose_mgdl"].ffill().bfill().to_numpy(dtype=float)
            g_series = np.clip(g_series, 40.0, 400.0)
        else:
            g_series = np.full(len(w_df), 140.0)

        # Derivatives
        if "glucose_roc" in w_df.columns:
            roc_series = w_df["glucose_roc"].fillna(0.0).to_numpy(dtype=float)
        else:
            roc_series = np.gradient(g_series, 5.0)

        if "glucose_acc" in w_df.columns:
            acc_series = w_df["glucose_acc"].fillna(0.0).to_numpy(dtype=float)
        else:
            acc_series = np.gradient(roc_series, 5.0)

        # HRV
        rmssd_series = (
            w_df["hrv_rmssd_ms"].fillna(38.0).to_numpy(dtype=float)
            if "hrv_rmssd_ms" in w_df.columns
            else np.full(len(w_df), 38.0)
        )
        sdnn_series = (
            w_df["hrv_sdnn_ms"].fillna(45.0).to_numpy(dtype=float)
            if "hrv_sdnn_ms" in w_df.columns
            else np.full(len(w_df), 45.0)
        )
        lf_hf_series = (
            w_df["hrv_lf_hf_ratio"].fillna(1.5).to_numpy(dtype=float)
            if "hrv_lf_hf_ratio" in w_df.columns
            else np.full(len(w_df), 1.5)
        )

        # Steps
        steps_5min = (
            w_df["steps_last_5min"].fillna(0).to_numpy(dtype=float)
            if "steps_last_5min" in w_df.columns
            else np.zeros(len(w_df))
        )

        # Timestamps
        last_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        if "timestamp" in w_df.columns and not w_df["timestamp"].isna().all():
            val = w_df["timestamp"].iloc[-1]
            if isinstance(val, (datetime, pd.Timestamp)):
                last_dt = val.to_pydatetime() if hasattr(val, "to_pydatetime") else val

        features: Dict[str, float] = {}

        # =============================================================
        # 1. EHR Static Features (23)
        # =============================================================
        features["age"] = float(ehr.get("age", 52))
        features["gender_encoded"] = float(ehr.get("gender_encoded", 1))
        features["bmi"] = float(ehr.get("bmi", 25.5))
        features["hba1c_latest"] = float(ehr.get("hba1c_latest", 8.0))
        features["hba1c_trend"] = float(ehr.get("hba1c_trend", 0.0))
        features["fasting_glucose_avg"] = float(ehr.get("fasting_glucose_avg", 155.0))
        features["egfr"] = float(ehr.get("egfr", 85.0))
        features["bp_sys"] = float(ehr.get("bp_sys", 130.0))
        features["diabetes_duration_years"] = float(ehr.get("diabetes_duration_years", 6))
        features["on_metformin"] = float(bool(ehr.get("on_metformin", True)))
        features["on_insulin"] = float(bool(ehr.get("on_insulin", False)))
        features["insulin_dose_units"] = float(ehr.get("insulin_dose_units", 0.0))
        features["on_sulphonylurea"] = float(bool(ehr.get("on_sulphonylurea", False)))
        features["on_sglt2"] = float(bool(ehr.get("on_sglt2", False)))
        features["has_hypertension"] = float(bool(ehr.get("has_hypertension", False)))
        features["has_dyslipidaemia"] = float(bool(ehr.get("has_dyslipidaemia", False)))
        features["has_neuropathy"] = float(bool(ehr.get("has_neuropathy", False)))
        features["has_retinopathy"] = float(bool(ehr.get("has_retinopathy", False)))
        features["bmi_category_encoded"] = float(ehr.get("bmi_category_encoded", 1))
        features["age_group_encoded"] = float(ehr.get("age_group_encoded", 1))
        features["ethnicity_encoded"] = float(ehr.get("ethnicity_encoded", 1))
        features["physical_activity_encoded"] = float(ehr.get("physical_activity_encoded", 1))
        features["dietary_compliance_score"] = float(ehr.get("dietary_compliance_score", 0.70))

        # =============================================================
        # 2. CGM Multi-Window Statistics (20)
        # =============================================================
        # 1 hour = 12 steps, 4 hours = 48 steps
        g_1h = g_series[-12:]
        g_4h = g_series[-48:]

        features["glucose_mean_1h"] = float(np.mean(g_1h))
        features["glucose_std_1h"] = float(np.std(g_1h)) if len(g_1h) > 1 else 0.0
        features["glucose_min_1h"] = float(np.min(g_1h))
        features["glucose_max_1h"] = float(np.max(g_1h))

        features["glucose_mean_4h"] = float(np.mean(g_4h))
        features["glucose_std_4h"] = float(np.std(g_4h)) if len(g_4h) > 1 else 0.0
        features["glucose_min_4h"] = float(np.min(g_4h))
        features["glucose_max_4h"] = float(np.max(g_4h))

        # 12h and 24h: use full available series or extrapolate from 4h window
        g_12h = g_series[-min(144, len(g_series)):]
        g_24h = g_series[-min(288, len(g_series)):]

        features["glucose_mean_12h"] = float(np.mean(g_12h))
        features["glucose_std_12h"] = float(np.std(g_12h)) if len(g_12h) > 1 else features["glucose_std_4h"]
        features["glucose_min_12h"] = float(np.min(g_12h))
        features["glucose_max_12h"] = float(np.max(g_12h))

        features["glucose_mean_24h"] = float(np.mean(g_24h))
        features["glucose_std_24h"] = float(np.std(g_24h)) if len(g_24h) > 1 else features["glucose_std_4h"]

        # ROC & ACC in short windows
        roc_30m = roc_series[-6:]
        features["glucose_roc_mean_30min"] = float(np.mean(roc_30m))
        features["glucose_roc_std_30min"] = float(np.std(roc_30m)) if len(roc_30m) > 1 else 0.0

        acc_15m = acc_series[-3:]
        features["glucose_acc_mean_15min"] = float(np.mean(acc_15m))

        features["glucose_percentile_10_4h"] = float(np.percentile(g_4h, 10))
        features["glucose_percentile_90_4h"] = float(np.percentile(g_4h, 90))
        features["glucose_iqr_4h"] = float(np.percentile(g_4h, 75) - np.percentile(g_4h, 25))

        # =============================================================
        # 3. CGM Derived Features (8)
        # =============================================================
        g_current = float(g_series[-1])
        roc_current = float(roc_series[-1])
        acc_current = float(acc_series[-1])

        features["glucose_current"] = g_current
        features["glucose_roc_current"] = roc_current
        features["glucose_acc_current"] = acc_current

        # Excursion counts in 4h window (transitions crossing 70 or 180)
        crossed_70 = np.sum((g_4h[:-1] >= 70) & (g_4h[1:] < 70)) + np.sum((g_4h[:-1] < 70) & (g_4h[1:] >= 70))
        crossed_180 = np.sum((g_4h[:-1] <= 180) & (g_4h[1:] > 180)) + np.sum((g_4h[:-1] > 180) & (g_4h[1:] <= 180))
        features["glucose_excursion_count_4h"] = float(crossed_70 + crossed_180)

        # Percentages [0.0, 100.0]
        features["glucose_above_180_pct_4h"] = float(np.mean(g_4h > 180.0) * 100.0)
        features["glucose_below_70_pct_4h"] = float(np.mean(g_4h < 70.0) * 100.0)

        # CV in 4h
        m_4h = features["glucose_mean_4h"]
        s_4h = features["glucose_std_4h"]
        features["glucose_coefficient_of_variation_4h"] = float((s_4h / (m_4h + 1e-6)) * 100.0)

        # Personal deviation
        if baseline_glucose is None:
            baseline_glucose = 28.7 * features["hba1c_latest"] - 46.7
        features["glucose_personal_deviation"] = float(g_current - baseline_glucose)

        # =============================================================
        # 4. HRV Features (12)
        # =============================================================
        rmssd_30m = rmssd_series[-6:]
        rmssd_4h = rmssd_series[-48:]

        rmssd_mean_30 = float(np.mean(rmssd_30m))
        rmssd_mean_4h = float(np.mean(rmssd_4h))

        features["hrv_rmssd_mean_30min"] = rmssd_mean_30
        features["hrv_rmssd_std_30min"] = float(np.std(rmssd_30m)) if len(rmssd_30m) > 1 else 0.0
        features["hrv_rmssd_min_30min"] = float(np.min(rmssd_30m))
        features["hrv_rmssd_mean_4h"] = rmssd_mean_4h

        # Trend over 4h: linear regression slope
        x_trend = np.arange(len(rmssd_4h), dtype=float)
        if len(rmssd_4h) > 1:
            slope = float(np.polyfit(x_trend, rmssd_4h, 1)[0])
        else:
            slope = 0.0
        features["hrv_rmssd_trend_4h"] = slope

        sdnn_1h = sdnn_series[-12:]
        features["hrv_sdnn_mean_1h"] = float(np.mean(sdnn_1h))
        features["hrv_sdnn_std_1h"] = float(np.std(sdnn_1h)) if len(sdnn_1h) > 1 else 0.0

        lf_hf_1h = lf_hf_series[-12:]
        features["hrv_lf_hf_mean_1h"] = float(np.mean(lf_hf_1h))

        # Suppression flag: acute drop > 40% from 4h average
        features["hrv_suppression_flag"] = 1.0 if rmssd_mean_30 < (0.60 * rmssd_mean_4h) else 0.0

        if baseline_hrv is None:
            baseline_hrv = 38.0
        features["hrv_personal_deviation"] = float(rmssd_series[-1] - baseline_hrv)

        # 30-min % change
        rmssd_prev = rmssd_series[-6] if len(rmssd_series) >= 6 else rmssd_series[0]
        features["hrv_rmssd_pct_change_30min"] = float(((rmssd_series[-1] - rmssd_prev) / (rmssd_prev + 1e-6)) * 100.0)

        max_rmssd_window = float(np.max(rmssd_series))
        features["hrv_recovery_score"] = float(np.clip(rmssd_series[-1] / (max_rmssd_window + 1e-6), 0.0, 1.0))

        # =============================================================
        # 5. Activity Features (8)
        # =============================================================
        steps_1h = float(np.sum(steps_5min[-12:]))
        steps_4h = float(np.sum(steps_5min[-48:]))
        # 24h: either use provided column or scale 4h
        if "steps_last_24h" in w_df.columns:
            steps_24h = float(w_df["steps_last_24h"].iloc[-1])
        else:
            steps_24h = steps_4h * 5.0

        features["steps_last_hour"] = steps_1h
        features["steps_last_4h"] = steps_4h
        features["steps_last_24h"] = steps_24h

        # Activity encoding: sleep: -1, resting: 0, walking: 1, exercise: 2
        act_map = {"sleep": -1.0, "resting": 0.0, "walking": 1.0, "exercise": 2.0}
        last_act = str(w_df["activity_type"].iloc[-1]).lower() if "activity_type" in w_df.columns else "resting"
        features["activity_type_encoded"] = act_map.get(last_act, 0.0)
        features["is_exercising"] = 1.0 if (last_act == "exercise" or steps_5min[-1] > 150) else 0.0
        features["is_sleeping"] = 1.0 if (last_act == "sleep") else 0.0

        # Time since exercise
        ex_steps = np.where(steps_5min > 120)[0]
        if len(ex_steps) > 0:
            last_ex_idx = ex_steps[-1]
            time_since_ex = float((len(steps_5min) - 1 - last_ex_idx) * 5.0)
        else:
            time_since_ex = 1440.0
        features["time_since_exercise_min"] = time_since_ex

        if "calories_last_hour" in w_df.columns:
            cal_1h = float(w_df["calories_last_hour"].iloc[-1])
        elif "calories_5min" in w_df.columns:
            cal_1h = float(np.sum(w_df["calories_5min"].iloc[-12:].fillna(1.0)))
        else:
            cal_1h = steps_1h * 0.04
        features["calories_last_hour"] = cal_1h

        # =============================================================
        # 6. Sleep Features (6)
        # =============================================================
        sleep_map = {"none": 0.0, "awake": 0.0, "light": 1.0, "deep": 2.0, "rem": 3.0}
        last_sleep = str(w_df["sleep_stage"].iloc[-1]).lower() if "sleep_stage" in w_df.columns else "none"
        features["sleep_stage_encoded"] = sleep_map.get(last_sleep, 0.0)

        sq = float(w_df["sleep_quality"].iloc[-1]) if "sleep_quality" in w_df.columns else 0.85
        if math.isnan(sq) or math.isinf(sq):
            sq = 0.85
        features["sleep_quality_current"] = sq

        # Check explicit columns or EHR profile overrides
        tot_sleep = (
            float(w_df["sleep_total_hours_last_night"].iloc[-1])
            if "sleep_total_hours_last_night" in w_df.columns
            else (float(ehr["sleep_total_hours_last_night"]) if "sleep_total_hours_last_night" in ehr else None)
        )
        eff_sleep = (
            float(w_df["sleep_efficiency_last_night"].iloc[-1])
            if "sleep_efficiency_last_night" in w_df.columns
            else (float(ehr["sleep_efficiency_last_night"]) if "sleep_efficiency_last_night" in ehr else None)
        )
        deep_sleep = (
            float(w_df["sleep_deep_pct_last_night"].iloc[-1])
            if "sleep_deep_pct_last_night" in w_df.columns
            else (float(ehr["sleep_deep_pct_last_night"]) if "sleep_deep_pct_last_night" in ehr else None)
        )

        # Check empirical telemetry across window if available
        if "sleep_stage" in w_df.columns:
            sleep_stages_lower = w_df["sleep_stage"].astype(str).str.lower()
            is_sleep_step = sleep_stages_lower.isin(["light", "deep", "rem"])
            if "activity_type" in w_df.columns:
                is_sleep_step = is_sleep_step | (w_df["activity_type"].astype(str).str.lower() == "sleep")
            n_sleep_steps = int(is_sleep_step.sum())
        else:
            is_sleep_step = pd.Series([False] * len(w_df))
            n_sleep_steps = 0

        # If a 24h window (288 steps) or full night (>36 sleep steps) is captured:
        if len(w_df) >= 288 and n_sleep_steps > 0:
            if tot_sleep is None:
                tot_sleep = round(float(n_sleep_steps * 5.0 / 60.0), 2)
            if deep_sleep is None:
                n_deep = int((sleep_stages_lower == "deep").sum())
                deep_sleep = round(float((n_deep / max(1, n_sleep_steps)) * 100.0), 1)
            if eff_sleep is None and "sleep_quality" in w_df.columns:
                eff_sleep = round(float(np.mean(w_df.loc[is_sleep_step, "sleep_quality"])), 3)

        # Dynamic physiological derivation from patient profile and wearable sleep quality
        age_val = features["age"]
        if tot_sleep is None:
            tot_sleep = round(float(np.clip(7.0 + 2.0 * (sq - 0.80) - 0.015 * (age_val - 50.0), 4.0, 10.0)), 2)
        if eff_sleep is None:
            eff_sleep = round(float(np.clip(0.76 + 0.16 * sq - 0.001 * (age_val - 50.0), 0.50, 0.98)), 3)
        if deep_sleep is None:
            deep_sleep = round(float(np.clip(21.0 + 8.0 * (sq - 0.80) - 0.10 * (age_val - 50.0), 5.0, 35.0)), 1)

        features["sleep_total_hours_last_night"] = float(tot_sleep)
        features["sleep_efficiency_last_night"] = float(eff_sleep)
        features["sleep_deep_pct_last_night"] = float(deep_sleep)

        cur_hour = last_dt.hour
        features["is_night_hours"] = 1.0 if cur_hour in (22, 23, 0, 1, 2, 3, 4, 5) else 0.0

        # =============================================================
        # 7. Temporal Features (6)
        # =============================================================
        features["hour_of_day"] = float(cur_hour)
        features["day_of_week"] = float(last_dt.weekday())
        features["is_weekend"] = 1.0 if last_dt.weekday() >= 5 else 0.0

        # Dynamic computation of time_since_last_meal_min:
        if "time_since_last_meal_min" in w_df.columns:
            features["time_since_last_meal_min"] = float(w_df["time_since_last_meal_min"].iloc[-1])
        elif "meal_event" in w_df.columns and w_df["meal_event"].any():
            meal_indices = np.where(w_df["meal_event"])[0]
            last_meal_idx = meal_indices[-1]
            features["time_since_last_meal_min"] = float((len(w_df) - 1 - last_meal_idx) * 5.0)
        else:
            # Estimate from glucose spike pattern (RoC > 1.2 mg/dL/min)
            spike_indices = np.where(roc_series > 1.2)[0]
            if len(spike_indices) > 0:
                last_spike_idx = spike_indices[-1]
                est_time = float((len(w_df) - 1 - last_spike_idx) * 5.0) + 15.0  # ~15 min ingestion lag
                features["time_since_last_meal_min"] = float(np.clip(est_time, 15.0, 720.0))
            else:
                # Circadian meal schedule: Breakfast ~08:00 (480m), Lunch ~13:00 (780m), Dinner ~20:00 (1200m)
                cur_min_of_day = cur_hour * 60.0 + last_dt.minute
                meal_schedule = [480.0, 780.0, 1200.0]
                past_meals = [m for m in meal_schedule if cur_min_of_day >= m]
                if past_meals:
                    elapsed = cur_min_of_day - past_meals[-1]
                else:
                    elapsed = cur_min_of_day + (1440.0 - 1200.0)
                features["time_since_last_meal_min"] = float(np.clip(elapsed, 10.0, 720.0))

        # Dynamic computation of time_since_last_insulin_min:
        if not features["on_insulin"]:
            features["time_since_last_insulin_min"] = 0.0
        elif "time_since_last_insulin_min" in w_df.columns:
            features["time_since_last_insulin_min"] = float(w_df["time_since_last_insulin_min"].iloc[-1])
        elif "insulin_event" in w_df.columns and w_df["insulin_event"].any():
            ins_indices = np.where(w_df["insulin_event"])[0]
            last_ins_idx = ins_indices[-1]
            features["time_since_last_insulin_min"] = float((len(w_df) - 1 - last_ins_idx) * 5.0)
        elif iob > 0.0 and features["insulin_dose_units"] > 0:
            # Hovorka pharmacokinetic decay inversion (tau ~ 65 min)
            frac_remaining = np.clip(iob / max(1.0, features["insulin_dose_units"]), 0.01, 0.99)
            est_time = -65.0 * np.log(frac_remaining)
            features["time_since_last_insulin_min"] = float(np.clip(est_time, 5.0, 360.0))
        else:
            # Prandial insulin administered with meals
            meal_time = features["time_since_last_meal_min"]
            features["time_since_last_insulin_min"] = float(meal_time) if meal_time <= 240.0 else 240.0

        features["circadian_phase"] = float((cur_hour + last_dt.minute / 60.0) / 24.0)

        # =============================================================
        # 8. Clinical Indices & PK (6)
        # =============================================================
        indices = compute_clinical_indices(g_24h, as_percentage=True)
        features["tir_24h"] = indices["tir_24h"]
        features["tar_24h"] = indices["tar_24h"]
        features["tbr_24h"] = indices["tbr_24h"]
        features["gmi"] = indices["gmi"]
        features["cv_glucose_24h"] = indices["cv_glucose_24h"]
        features["io_board"] = float(iob)

        # -------------------------------------------------------------
        # Clean up and Guarantee Exactly 89 Keys in Order
        # -------------------------------------------------------------
        ordered_features: Dict[str, float] = {}
        for col in FEATURE_COLUMNS:
            val = features.get(col, 0.0)
            if math.isnan(val) or math.isinf(val):
                val = 0.0
            ordered_features[col] = float(val)

        return ordered_features

    def compute_batch_features(
        self,
        patient_df: pd.DataFrame,
        ehr_features: Optional[Dict[str, Any]] = None,
        window_size: int = 48,
    ) -> pd.DataFrame:
        """
        Computes 89-feature vectors across time for an entire patient dataset.
        """
        records = []
        n = len(patient_df)
        for i in range(window_size, n):
            window_slice = patient_df.iloc[i - window_size : i]
            feats = self.compute_features(window_slice, ehr_features)
            feats["timestamp"] = patient_df.iloc[i]["timestamp"] if "timestamp" in patient_df.columns else i
            records.append(feats)

        return pd.DataFrame(records)
