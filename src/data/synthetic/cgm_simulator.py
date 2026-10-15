"""
src/data/synthetic/cgm_simulator.py

Physiological 5-minute continuous glucose monitoring (CGM) simulator
with diurnal circadian rhythms, dawn phenomenon, meal absorption dynamics,
exercise excursions, insulin pharmacokinetics, and CGM sensor noise model.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class MealEvent:
    timestamp: datetime
    carbs_grams: float
    meal_type: str  # "breakfast", "lunch", "dinner", "snack"


@dataclass
class ExerciseEvent:
    timestamp: datetime
    duration_min: int
    intensity: str  # "light", "moderate", "vigorous"


@dataclass
class InsulinEvent:
    timestamp: datetime
    dose_units: float
    insulin_type: str  # "rapid_acting", "long_acting"


class CGMNoiseModel:
    """
    Simulates real-world CGM sensor physics and artifacts:
    - 7-15 min physiological interstitial lag
    - AR(1) autocorrelated Gaussian noise (MARD 5-10%, phi=0.8)
    - 14-day calibration drift factor
    - 5% random dropouts + physical exercise sensor displacement
    - Integer quantization and physiological clamping [40.0, 400.0] mg/dL
    """

    def __init__(
        self,
        m_ard: float = 0.07,
        mard: Optional[float] = None,
        ar1_phi: float = 0.80,
        lag_min_range: Tuple[float, float] = (7.0, 15.0),
        dropout_prob: float = 0.05,
        sensor_age_days: float = 0.0,
        seed: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        actual_seed = seed if seed is not None else random_seed
        self.mard = mard if mard is not None else m_ard
        self.ar1_phi = ar1_phi
        self.lag_min_range = lag_min_range
        self.dropout_prob = dropout_prob
        self.sensor_age_days = sensor_age_days
        self.rng = np.random.default_rng(actual_seed)

    def apply(
        self,
        clean_glucose: pd.Series,
        exercise_flags: Optional[pd.Series] = None,
        return_quality: bool = False,
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Applies sensor noise, drift, lag, and dropouts to clean glucose series.

        Args:
            clean_glucose: Clean blood glucose pd.Series.
            exercise_flags: Optional boolean series of exercise periods.
            return_quality: If True, returns (noisy_series, quality_series);
                            if False, returns noisy_series.
        """
        n = len(clean_glucose)
        if n == 0:
            empty_s = pd.Series([], dtype=float, index=clean_glucose.index)
            if return_quality:
                return empty_s, pd.Series([], dtype=object, index=clean_glucose.index)
            return empty_s

        g_vals = clean_glucose.values.astype(float).copy()
        original_missing = np.isnan(g_vals)

        # Impute missing values on scratch buffer for physical continuous calculations
        imputed_series = clean_glucose.ffill().bfill()
        if imputed_series.isna().all():
            g_physical = np.full(n, 120.0, dtype=float)
        else:
            g_physical = imputed_series.fillna(120.0).values.astype(float)

        # 1. Interstitial Lag: 7-15 min physiological lag
        lag_minutes = float(self.rng.uniform(*self.lag_min_range))
        lag_steps = lag_minutes / 5.0
        roc = np.zeros(n)
        if n > 1:
            roc[1:] = (g_physical[1:] - g_physical[:-1])
        lagged_g = g_physical - (lag_steps * roc)
        # Safeguard against negative interstitial glucose during rapid spikes
        lagged_g = np.maximum(lagged_g, 20.0)

        # 2. AR(1) Autocorrelated Gaussian Noise with NaN Immunity
        sigma = np.maximum(self.mard * lagged_g, 1.0)
        white_noise = self.rng.normal(0.0, 1.0, size=n) * sigma
        noise = np.zeros(n)
        for t in range(n):
            if np.isnan(white_noise[t]):
                noise[t] = noise[t - 1] if (t > 0 and not np.isnan(noise[t - 1])) else 0.0
            elif t == 0:
                noise[t] = white_noise[t]
            else:
                prev = noise[t - 1] if not np.isnan(noise[t - 1]) else 0.0
                noise[t] = self.ar1_phi * prev + (1.0 - self.ar1_phi) * white_noise[t]

        # 3. 14-Day Calibration Sensor Drift (C1 Smoothstep Boundary Smoothing)
        day_indices = (np.arange(n) / 288.0) + self.sensor_age_days
        cycle = 14.0
        w = 0.5  # 12-hour transition window across sensor replacement boundary
        u = day_indices % cycle
        d = (day_indices + w) % cycle
        in_transition = d < (2.0 * w)
        alpha = d / (2.0 * w)
        s_alpha = 3.0 * (alpha ** 2) - 2.0 * (alpha ** 3)
        val_transition = (1.0 - s_alpha) * (cycle - w) + s_alpha * w
        day_in_sensor = np.where(in_transition, val_transition, u)
        drift_factor = 1.0 + 0.005 * day_in_sensor

        noisy_g = (lagged_g + noise) * drift_factor

        # 4. Missingness & Dropouts
        quality = np.array(["good"] * n, dtype=object)
        high_noise_mask = np.abs(noise) > (1.5 * sigma)
        quality[high_noise_mask] = "noisy"

        random_drops = self.rng.random(n) < self.dropout_prob
        exercise_drops = np.zeros(n, dtype=bool)
        if exercise_flags is not None:
            ex_indices = np.where(exercise_flags.values)[0]
            for idx in ex_indices:
                if self.rng.random() < 0.35:
                    gap_len = int(self.rng.integers(3, 7))
                    exercise_drops[idx : min(n, idx + gap_len)] = True

        missing_mask = random_drops | exercise_drops | original_missing
        quality[missing_mask] = "missing"
        noisy_g[missing_mask] = np.nan

        # 5. Clamping to [40.0, 400.0]
        valid_mask = ~np.isnan(noisy_g)
        noisy_g[valid_mask] = np.clip(np.round(noisy_g[valid_mask]), 40.0, 400.0)

        noisy_series = pd.Series(noisy_g, index=clean_glucose.index)
        quality_series = pd.Series(quality, index=clean_glucose.index)

        if return_quality:
            return noisy_series, quality_series
        return noisy_series


class CGMSimulator:
    """
    Simulates 5-minute continuous glucose readings for Type 2 Diabetes patients
    incorporating physiological baselines, diurnal rhythm, dawn phenomenon,
    meals, exercise, insulin pharmacokinetics, and sensor artifacts.
    """

    def __init__(
        self,
        patient_ehr: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        actual_seed = seed if seed is not None else (random_seed if random_seed is not None else 42)
        self.patient_ehr = patient_ehr or {}
        self.rng = np.random.default_rng(actual_seed)
        self.noise_model = CGMNoiseModel(seed=actual_seed)

    def generate(
        self,
        patient_ehr: Optional[Dict[str, Any]] = None,
        duration_days: int = 30,
        start_time: Optional[datetime] = None,
        add_noise: bool = True,
    ) -> pd.DataFrame:
        """
        Generates 5-minute sampled CGM time series for the specified duration.
        """
        ehr = dict(self.patient_ehr)
        if patient_ehr is not None:
            ehr.update(patient_ehr)

        if start_time is None:
            start_time = datetime(2026, 10, 1, 0, 0, 0)

        n_steps = duration_days * 288
        timestamps = [start_time + timedelta(minutes=5 * i) for i in range(n_steps)]

        # 1. Baseline glucose and patient variability
        hba1c = float(ehr.get("hba1c_latest", 8.0))
        eag_baseline = 28.7 * hba1c - 46.7
        fasting_avg = float(ehr.get("fasting_glucose_avg", eag_baseline))
        base_glucose = 0.6 * eag_baseline + 0.4 * fasting_avg

        bmi = float(ehr.get("bmi", 25.5))
        volatility_sigma = 6.0 + 2.5 * max(0.0, hba1c - 6.5) + 0.3 * max(0.0, bmi - 23.0)

        # 2. Circadian Rhythm & Dawn Phenomenon
        circadian = np.zeros(n_steps)
        for i, dt in enumerate(timestamps):
            hour = dt.hour + dt.minute / 60.0
            diurnal = 8.0 * np.sin(2.0 * np.pi * (hour - 14.0) / 24.0)
            dawn_amp = 16.0 + 3.0 * max(0.0, hba1c - 7.0)
            dawn = dawn_amp * np.exp(-((hour - 6.0) ** 2) / (2.0 * (1.2 ** 2)))
            circadian[i] = diurnal + dawn

        # 3. Schedule Meals & Excursions
        meal_curve = np.zeros(n_steps)
        meal_flags = np.zeros(n_steps, dtype=bool)
        exercise_curve = np.zeros(n_steps)
        exercise_flags = np.zeros(n_steps, dtype=bool)
        insulin_curve = np.zeros(n_steps)
        insulin_flags = np.zeros(n_steps, dtype=bool)

        on_insulin = bool(ehr.get("on_insulin", False))
        tdd = float(ehr.get("insulin_dose_units", 30.0 if on_insulin else 0.0))
        isf = 1800.0 / max(10.0, tdd) if on_insulin else 50.0

        for day in range(duration_days):
            day_offset = day * 288

            # Daily meals (Breakfast ~08:00, Lunch ~13:00, Dinner ~20:00, Snack ~16:30)
            meals = [
                (int(self.rng.normal(8.0 * 12, 3)), float(self.rng.uniform(40, 70))),
                (int(self.rng.normal(13.0 * 12, 4)), float(self.rng.uniform(50, 85))),
                (int(self.rng.normal(20.0 * 12, 4)), float(self.rng.uniform(60, 105))),
            ]
            if self.rng.random() < 0.40:
                meals.append((int(self.rng.normal(16.5 * 12, 3)), float(self.rng.uniform(20, 40))))

            for meal_step, carbs in meals:
                idx = day_offset + meal_step
                if 0 <= idx < n_steps:
                    meal_flags[idx] = True
                    k_carb = 1.3 + 0.1 * (bmi / 22.0)
                    a_meal = carbs * k_carb
                    tau_peak = 12.0  # 60 min
                    for s in range(48):
                        if idx + s < n_steps:
                            t_s = float(s)
                            if t_s > 0:
                                meal_curve[idx + s] += a_meal * (t_s / tau_peak) * np.exp(1.0 - t_s / tau_peak)

                    if on_insulin:
                        insulin_flags[idx] = True
                        dose = (carbs / 10.0) * float(self.rng.uniform(0.8, 1.2))
                        tau1 = 0.35 * 4.0 / np.log(2) * 12
                        tau2 = (4.0 - 0.35 * 4.0) / np.log(2) * 12
                        for s in range(48):
                            if idx + s < n_steps:
                                t_s = float(s)
                                iob = dose * (0.7 * np.exp(-t_s / tau1) + 0.3 * np.exp(-t_s / tau2))
                                insulin_curve[idx + s] -= (iob * isf * 0.15)

            # Daily exercise session
            act_level = int(ehr.get("physical_activity_encoded", 1))
            ex_prob = [0.15, 0.40, 0.70, 0.90][min(3, max(0, act_level))]
            if self.rng.random() < ex_prob:
                ex_hour = float(self.rng.choice([7.0, 17.5, 18.5]))
                ex_step = day_offset + int(ex_hour * 12)
                ex_duration = int(self.rng.integers(6, 12))
                if 0 <= ex_step < n_steps:
                    for s in range(ex_duration):
                        if ex_step + s < n_steps:
                            exercise_flags[ex_step + s] = True
                            exercise_curve[ex_step + s] -= (3.0 * (s + 1))
                    for s in range(ex_duration, ex_duration + 24):
                        if ex_step + s < n_steps:
                            exercise_curve[ex_step + s] -= 10.0 * np.exp(-(s - ex_duration) / 12.0)

        # 4. Stochastic Ornstein-Uhlenbeck walk
        stochastic_walk = np.zeros(n_steps)
        theta = 0.15
        for t in range(n_steps):
            if t == 0:
                stochastic_walk[t] = self.rng.normal(0, volatility_sigma)
            else:
                drift = -theta * stochastic_walk[t - 1]
                stochastic_walk[t] = stochastic_walk[t - 1] + drift + self.rng.normal(0, volatility_sigma * 0.4)

        clean_glucose = (
            base_glucose
            + circadian
            + meal_curve
            + exercise_curve
            + insulin_curve
            + stochastic_walk
        )
        clean_glucose = np.clip(clean_glucose, 45.0, 395.0)

        # 5. Apply Noise Model
        clean_series = pd.Series(clean_glucose, index=timestamps)
        ex_series = pd.Series(exercise_flags, index=timestamps)

        if add_noise:
            final_glucose, quality_series = self.noise_model.apply(
                clean_series, ex_series, return_quality=True
            )
            imputed_g = final_glucose.ffill().bfill().fillna(base_glucose).to_numpy()
        else:
            final_glucose = clean_series
            quality_series = pd.Series(["good"] * n_steps, index=timestamps)
            imputed_g = clean_glucose

        # 6. Compute Derivatives
        roc = np.zeros(n_steps)
        acc = np.zeros(n_steps)
        if n_steps > 1:
            roc[1:] = (imputed_g[1:] - imputed_g[:-1]) / 5.0
            acc[1:] = (roc[1:] - roc[:-1]) / 5.0

        pid = str(ehr.get("patient_id", "PAT-SYNTH-001"))

        df = pd.DataFrame({
            "patient_id": pid,
            "timestamp": timestamps,
            "glucose_mgdl": final_glucose.values,
            "glucose_roc": np.round(roc, 3),
            "glucose_acc": np.round(acc, 4),
            "meal_event": meal_flags,
            "exercise_event": exercise_flags,
            "insulin_event": insulin_flags,
            "data_quality": quality_series.values,
        })
        return df
