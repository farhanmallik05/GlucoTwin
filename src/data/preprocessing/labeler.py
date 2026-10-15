"""
src/data/preprocessing/labeler.py

Adverse glycemic event ground-truth labeler.
Labels each timestep with forward 120-minute horizon (24 steps at 5-min sampling)
for adverse excursions:
- Hypoglycemia: G < 70.0 mg/dL
- Severe Hyperglycemia: G > 250.0 mg/dL
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


def label_adverse_events(
    cgm_df: pd.DataFrame,
    glucose_col: str = "glucose_mgdl",
    horizon_minutes: int = 120,
    step_minutes: int = 5,
    low_threshold: float = 70.0,
    high_threshold: float = 250.0,
    drop_trailing_unlabeled: bool = False,
) -> pd.DataFrame:
    """
    Labels each row with forward adverse event outcome within horizon_minutes.

    Args:
        cgm_df: DataFrame containing continuous glucose readings.
        glucose_col: Column name containing glucose readings in mg/dL.
        horizon_minutes: Forward lookahead horizon (default 120 min).
        step_minutes: Timestep interval in minutes (default 5 min).
        low_threshold: Hypoglycemia cutoff (default 70.0 mg/dL).
        high_threshold: Severe hyperglycemia cutoff (default 250.0 mg/dL).
        drop_trailing_unlabeled: If True, drops trailing rows lacking full horizon.

    Returns:
        DataFrame copy with added columns:
        - 'adverse_event': float/int (0, 1, or np.nan for trailing rows)
        - 'minutes_to_event': float (lead time to earliest excursion) or np.nan
        - 'event_type': str ('hypoglycaemia', 'hyperglycaemia', 'none', 'unlabeled')
    """
    df = cgm_df.copy()
    n = len(df)
    horizon_steps = horizon_minutes // step_minutes  # 24 steps

    glucose = df[glucose_col].values.astype(float)

    adverse_event = np.full(n, np.nan)
    minutes_to_event = np.full(n, np.nan)
    event_type = np.array(["unlabeled"] * n, dtype=object)

    for i in range(n - horizon_steps):
        # Lookahead window of next horizon_steps: i+1 to i + horizon_steps
        future_window = glucose[i + 1 : i + 1 + horizon_steps]
        
        # Valid non-nan readings
        valid_indices = np.where(~np.isnan(future_window))[0]
        if len(valid_indices) == 0:
            continue

        # Find first violation
        earliest_step = None
        earliest_type = "none"

        for step_idx in valid_indices:
            val = future_window[step_idx]
            if val < low_threshold:
                earliest_step = step_idx + 1  # 1-based step ahead (1 to 24)
                earliest_type = "hypoglycaemia"
                break
            elif val > high_threshold:
                earliest_step = step_idx + 1
                earliest_type = "hyperglycaemia"
                break

        if earliest_step is not None:
            adverse_event[i] = 1
            minutes_to_event[i] = earliest_step * step_minutes
            event_type[i] = earliest_type
        else:
            adverse_event[i] = 0
            event_type[i] = "none"

    df["adverse_event"] = adverse_event
    df["minutes_to_event"] = minutes_to_event
    df["event_type"] = event_type

    if drop_trailing_unlabeled:
        df = df.iloc[: n - horizon_steps].copy()

    return df
