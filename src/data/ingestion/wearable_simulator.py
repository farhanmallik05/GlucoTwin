"""
src/data/ingestion/wearable_simulator.py

Multi-modal wearable sensor simulator that integrates CGM, autonomic HRV,
step counter, and sleep architecture into a unified synchronized time-series.
Defines the canonical WearableReading contract.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.synthetic.hrv_simulator import HRVSimulator


@dataclass
class WearableReading:
    """
    Standardized multi-modal 5-minute sensor reading.
    Serves as the canonical interface contract for M1, M2, M4, and E2E tiers.
    """
    patient_id: str
    timestamp: datetime

    # CGM Channels
    glucose_mgdl: float           # Interstitial glucose [40.0, 400.0] mg/dL
    glucose_roc: float            # 1st derivative: rate of change (mg/dL/min)
    glucose_acc: float = 0.0      # 2nd derivative: acceleration (mg/dL/min²)

    # Autonomic / Heart Rate Channels
    heart_rate_bpm: float = 75.0  # Heart rate [40.0, 200.0] bpm
    hrv_rmssd_ms: float = 38.0    # RMSSD [5.0, 120.0] ms
    hrv_sdnn_ms: float = 45.0     # SDNN [10.0, 150.0] ms
    hrv_lf_hf_ratio: float = 1.5  # Sympathovagal ratio [0.1, 10.0]

    # Activity Channels
    steps_last_5min: int = 0      # Steps in 5-min window [0, 1000]
    steps_last_hour: int = 0      # Rolling 1-hour step count [0, 12000]
    activity_type: str = "resting"# "resting" | "walking" | "exercise" | "sleep"
    calories_5min: float = 0.0    # Energy expenditure in kcal

    # Sleep Channels
    sleep_stage: str = "none"     # "awake" | "light" | "deep" | "rem" | "none"
    sleep_quality: float = 0.85   # Quality score [0.0, 1.0]

    # Metadata & Provenance
    data_quality: str = "good"    # "good" | "noisy" | "missing"
    source: str = "cgm_sim"       # "cgm_sim" | "ohio" | "real_device"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes reading into JSON-compatible dictionary."""
        d = asdict(self)
        if isinstance(d["timestamp"], datetime):
            d["timestamp"] = d["timestamp"].isoformat()
        return d


class WearableSimulator:
    """
    Multi-modal wearable sensor simulator that integrates CGM, autonomic HRV,
    step counter, and sleep hypnogram data into a unified synchronized time-series.
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
        self.hrv_sim = HRVSimulator(patient_ehr=self.patient_ehr, random_seed=actual_seed)

    def generate(
        self,
        patient_ehr: Optional[Dict[str, Any]] = None,
        cgm_df: Optional[pd.DataFrame] = None,
        duration_days: int = 30,
        start_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generates or augments CGM DataFrame with synchronized HRV, activity, and sleep.
        """
        ehr = dict(self.patient_ehr)
        if patient_ehr is not None:
            ehr.update(patient_ehr)

        pid = ehr.get("patient_id", "PAT-001")

        # 1. Base CGM time series
        if cgm_df is not None and not cgm_df.empty:
            df = cgm_df.copy()
            n_steps = len(df)
            if "timestamp" in df.columns:
                timestamps = pd.to_datetime(df["timestamp"]).tolist()
            else:
                s_time = start_time or datetime(2026, 10, 1, 0, 0, 0)
                timestamps = [s_time + timedelta(minutes=5 * i) for i in range(n_steps)]
                df["timestamp"] = timestamps
        else:
            # Import CGMSimulator dynamically if needed
            from src.data.synthetic.cgm_simulator import CGMSimulator
            cgm_sim = CGMSimulator(patient_ehr=ehr, random_seed=int(self.rng.integers(0, 100000)))
            df = cgm_sim.generate(patient_ehr=ehr, duration_days=duration_days, start_time=start_time)
            n_steps = len(df)
            timestamps = pd.to_datetime(df["timestamp"]).tolist()

        # 2. Activity & Sleep state simulation
        activity_types = []
        steps_5m = []
        calories_5m = []
        sleep_stages = []
        sleep_qualities = []
        sleep_flags = np.zeros(n_steps, dtype=bool)

        weight_kg = float(ehr.get("bmi", 25.0)) * ((float(ehr.get("height_cm", 168.0)) / 100.0) ** 2)
        weight_kg = max(45.0, min(120.0, weight_kg))

        for i, dt in enumerate(timestamps):
            hour = dt.hour + dt.minute / 60.0

            # Sleep window: 23:00 to 07:00
            if hour >= 23.0 or hour < 7.0:
                sleep_flags[i] = True
                activity_types.append("sleep")
                steps_5m.append(int(self.rng.choice([0, 0, 0, 5], p=[0.85, 0.10, 0.04, 0.01])))
                calories_5m.append(round(weight_kg * 0.015, 2))

                # Sleep stage cycles (90 min cycles)
                cycle_progress = ((hour - 23.0 if hour >= 23.0 else hour + 1.0) * 60.0) % 90.0
                if cycle_progress < 15.0:
                    stage = "light"
                elif cycle_progress < 50.0:
                    stage = "deep"
                elif cycle_progress < 75.0:
                    stage = "rem"
                else:
                    stage = "light"
                sleep_stages.append(stage)
                sleep_qualities.append(0.88)
            else:
                sleep_stages.append("awake")
                sleep_qualities.append(0.85)

                # Daytime activity
                # Check if cgm_df had exercise event
                had_ex = bool(df["exercise_event"].iloc[i]) if "exercise_event" in df.columns else False
                if had_ex:
                    activity_types.append("exercise")
                    st = int(self.rng.integers(350, 650))
                    steps_5m.append(st)
                    calories_5m.append(round(st * 0.05, 2))
                else:
                    # Probabilistic daytime movement
                    r = self.rng.random()
                    if r < 0.70:
                        activity_types.append("resting")
                        st = int(self.rng.choice([0, 10, 25, 45]))
                        steps_5m.append(st)
                        calories_5m.append(round(st * 0.03 + 1.0, 2))
                    else:
                        activity_types.append("walking")
                        st = int(self.rng.integers(80, 220))
                        steps_5m.append(st)
                        calories_5m.append(round(st * 0.04, 2))

        # 3. Rolling 1-hour steps
        steps_arr = np.array(steps_5m, dtype=int)
        steps_1h = np.zeros(n_steps, dtype=int)
        for i in range(n_steps):
            start_idx = max(0, i - 11)
            steps_1h[i] = int(np.sum(steps_arr[start_idx : i + 1]))

        # 4. HRV Simulation coupled with glucose and sleep
        hrv_df = self.hrv_sim.simulate(df, sleep_flags=sleep_flags)

        # 5. Assemble unified DataFrame
        df["patient_id"] = pid
        df["heart_rate_bpm"] = hrv_df["heart_rate_bpm"]
        df["hrv_rmssd_ms"] = hrv_df["hrv_rmssd_ms"]
        df["hrv_sdnn_ms"] = hrv_df["hrv_sdnn_ms"]
        df["hrv_lf_hf_ratio"] = hrv_df["hrv_lf_hf_ratio"]
        df["steps_last_5min"] = steps_arr
        df["steps_last_hour"] = steps_1h
        df["activity_type"] = activity_types
        df["calories_5min"] = calories_5m
        df["sleep_stage"] = sleep_stages
        df["sleep_quality"] = sleep_qualities

        if "glucose_roc" not in df.columns:
            g = df["glucose_mgdl"].values.astype(float)
            roc = np.zeros(n_steps)
            if n_steps > 1:
                roc[1:] = (g[1:] - g[:-1]) / 5.0
            df["glucose_roc"] = np.round(roc, 3)

        if "glucose_acc" not in df.columns:
            r = df["glucose_roc"].values.astype(float)
            acc = np.zeros(n_steps)
            if n_steps > 1:
                acc[1:] = (r[1:] - r[:-1]) / 5.0
            df["glucose_acc"] = np.round(acc, 4)

        if "data_quality" not in df.columns:
            df["data_quality"] = "good"
        if "source" not in df.columns:
            df["source"] = "cgm_sim"

        return df

    def to_readings(
        self,
        df: pd.DataFrame,
        patient_id: Optional[str] = None,
    ) -> List[WearableReading]:
        """Converts DataFrame rows into typed WearableReading instances."""
        readings = []
        for _, row in df.iterrows():
            pid = patient_id or str(row.get("patient_id", "PAT-001"))
            ts = row.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()

            reading = WearableReading(
                patient_id=pid,
                timestamp=ts,
                glucose_mgdl=float(row.get("glucose_mgdl", 140.0)),
                glucose_roc=float(row.get("glucose_roc", 0.0)),
                glucose_acc=float(row.get("glucose_acc", 0.0)),
                heart_rate_bpm=float(row.get("heart_rate_bpm", 72.0)),
                hrv_rmssd_ms=float(row.get("hrv_rmssd_ms", 38.0)),
                hrv_sdnn_ms=float(row.get("hrv_sdnn_ms", 45.0)),
                hrv_lf_hf_ratio=float(row.get("hrv_lf_hf_ratio", 1.5)),
                steps_last_5min=int(row.get("steps_last_5min", 0)),
                steps_last_hour=int(row.get("steps_last_hour", 0)),
                activity_type=str(row.get("activity_type", "resting")),
                calories_5min=float(row.get("calories_5min", 0.0)),
                sleep_stage=str(row.get("sleep_stage", "none")),
                sleep_quality=float(row.get("sleep_quality", 0.85)),
                data_quality=str(row.get("data_quality", "good")),
                source=str(row.get("source", "cgm_sim")),
            )
            readings.append(reading)
        return readings
