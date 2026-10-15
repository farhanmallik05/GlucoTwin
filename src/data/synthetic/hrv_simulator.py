"""
src/data/synthetic/hrv_simulator.py

Autonomic nervous system HRV simulator coupling continuous glucose variations
with autonomic sympathetic/parasympathetic tone:
- Hypoglycemia (<80 mg/dL): acute sympathetic activation triggers 30-50% RMSSD drop, HR increase (+10-20 bpm), LF/HF spike (2.5-6.0).
- Hyperglycemia (>200 mg/dL): moderate autonomic suppression with 15-25% RMSSD drop.
- Sleep: parasympathetic dominance with peak RMSSD and nadir HR.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


class HRVSimulator:
    """
    Simulates autonomic HRV response (RMSSD, SDNN, LF/HF ratio, Heart Rate)
    coupled with glucose levels and circadian state.
    """

    def __init__(
        self,
        patient_ehr: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = 42,
    ):
        self.patient_ehr = patient_ehr or {}
        self.rng = np.random.default_rng(random_seed)

        # Baseline physiology derived from EHR
        age = float(self.patient_ehr.get("age", 52))
        bmi = float(self.patient_ehr.get("bmi", 26.0))
        hba1c = float(self.patient_ehr.get("hba1c_latest", 8.0))
        activity_level = int(self.patient_ehr.get("physical_activity_encoded", 1))

        # Baseline resting HR [60, 85] bpm
        self.baseline_hr = 68.0 + 0.15 * (age - 40.0) + 0.5 * (bmi - 23.0) - 2.5 * activity_level
        self.baseline_hr = float(np.clip(self.baseline_hr, 55.0, 95.0))

        # Baseline RMSSD [25, 65] ms
        self.baseline_rmssd = 45.0 - 0.35 * (age - 35.0) - 0.4 * (bmi - 22.0) - 1.2 * max(0.0, hba1c - 6.5) + 3.0 * activity_level
        self.baseline_rmssd = float(np.clip(self.baseline_rmssd, 20.0, 75.0))

        # Baseline SDNN [35, 80] ms
        self.baseline_sdnn = self.baseline_rmssd * 1.25

    def simulate(
        self,
        cgm_df: pd.DataFrame,
        sleep_flags: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Simulates autonomic HRV channels matching cgm_df timesteps.

        Args:
            cgm_df: DataFrame containing at least 'glucose_mgdl'.
            sleep_flags: Optional boolean array indicating sleep state.

        Returns:
            pd.DataFrame with 'heart_rate_bpm', 'hrv_rmssd_ms', 'hrv_sdnn_ms', 'hrv_lf_hf_ratio'.
        """
        n = len(cgm_df)
        glucose = cgm_df["glucose_mgdl"].values.astype(float)

        hr = np.zeros(n)
        rmssd = np.zeros(n)
        sdnn = np.zeros(n)
        lf_hf = np.zeros(n)

        for i in range(n):
            g = glucose[i] if not np.isnan(glucose[i]) else 140.0
            is_sleep = bool(sleep_flags[i]) if sleep_flags is not None else False

            # Physiological coupling with glucose
            if g < 80.0:
                # Acute hypoglycemia: severe sympathetic surge, RMSSD drops 30-50%
                severity = np.clip((80.0 - g) / 40.0, 0.0, 1.0)
                rmssd_mult = 1.0 - (0.30 + 0.20 * severity)  # 0.50 to 0.70 of baseline
                hr_add = 12.0 + 15.0 * severity
                lf_hf_val = 2.5 + 3.0 * severity
            elif g > 200.0:
                # Hyperglycemia: moderate autonomic suppression
                severity = np.clip((g - 200.0) / 150.0, 0.0, 1.0)
                rmssd_mult = 1.0 - (0.15 + 0.15 * severity)
                hr_add = 4.0 + 6.0 * severity
                lf_hf_val = 1.8 + 1.2 * severity
            else:
                rmssd_mult = 1.0
                hr_add = 0.0
                lf_hf_val = 1.4

            # Sleep effects
            if is_sleep:
                rmssd_mult *= 1.35
                hr_add -= 10.0
                lf_hf_val *= 0.65

            # Noise and bounds
            noise_rmssd = float(self.rng.normal(0, 1.5))
            noise_hr = float(self.rng.normal(0, 1.5))
            noise_lf_hf = float(self.rng.normal(0, 0.1))

            rmssd[i] = float(np.clip(self.baseline_rmssd * rmssd_mult + noise_rmssd, 5.0, 120.0))
            sdnn[i] = float(np.clip(rmssd[i] * 1.25 + self.rng.normal(0, 2.0), 10.0, 150.0))
            hr[i] = float(np.clip(self.baseline_hr + hr_add + noise_hr, 42.0, 195.0))
            lf_hf[i] = float(np.clip(lf_hf_val + noise_lf_hf, 0.1, 10.0))

        return pd.DataFrame({
            "heart_rate_bpm": np.round(hr, 1),
            "hrv_rmssd_ms": np.round(rmssd, 1),
            "hrv_sdnn_ms": np.round(sdnn, 1),
            "hrv_lf_hf_ratio": np.round(lf_hf, 2),
        })
