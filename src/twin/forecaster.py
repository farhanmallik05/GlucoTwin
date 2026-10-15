"""
src/twin/forecaster.py
Trajectory Forecaster projecting glucose dynamics over 120-minute horizon (24 steps of 5 min).
Models momentum decay, active IOB insulin-mediated disposal, and expanding confidence cone.
"""

from typing import List, Dict, Any
import numpy as np


class GlucoseForecaster:
    """
    Physiological trajectory forecaster for Type 2 Diabetes digital twins.
    Projects future blood glucose levels at 5-minute resolution across a configurable horizon.
    """

    def __init__(self, step_minutes: int = 5):
        self.step_minutes = step_minutes

    def forecast(
        self,
        current_glucose: float,
        current_roc: float,
        iob: float = 0.0,
        isf: float = 50.0,
        horizon_minutes: int = 120
    ) -> List[Dict[str, Any]]:
        """
        Compute projected trajectory with dynamic confidence intervals.
        
        Args:
            current_glucose: baseline glucose at t=0 (mg/dL)
            current_roc: rate of change (mg/dL/min)
            iob: insulin on board (Units)
            isf: insulin sensitivity factor (mg/dL/Unit)
            horizon_minutes: forecast horizon (default 120 minutes)
            
        Returns:
            List of 24 step dictionaries with projected glucose and confidence bounds.
        """
        num_steps = max(1, int(horizon_minutes // self.step_minutes))
        trajectory: List[Dict[str, Any]] = []

        g_projected = float(current_glucose)
        roc = float(current_roc)

        # Dampening factor for velocity over time (momentum dissipation)
        roc_decay = 0.94
        
        # Incremental insulin disposal rate per 5-min step
        insulin_effect_per_step = (iob * isf / 24.0) if iob > 0 else 0.0

        for step_idx in range(1, num_steps + 1):
            minutes_ahead = step_idx * self.step_minutes

            # Forward integration with momentum decay and active insulin pull
            delta_g = (roc * self.step_minutes) - insulin_effect_per_step
            g_projected += delta_g
            roc *= roc_decay

            # Physiological clamping to valid biological limits [40.0, 400.0]
            clamped_projected = float(np.clip(g_projected, 40.0, 400.0))

            # Expanding uncertainty cone (sigma grows with sqrt of time)
            uncertainty = 4.0 + 3.2 * np.sqrt(step_idx)
            g_lower = float(np.clip(clamped_projected - uncertainty, 40.0, 400.0))
            g_upper = float(np.clip(clamped_projected + uncertainty, 40.0, 400.0))

            # Guarantee ordering: lower <= projected <= upper
            g_lower = min(g_lower, clamped_projected)
            g_upper = max(g_upper, clamped_projected)

            trajectory.append({
                "step": step_idx,
                "minutes_ahead": minutes_ahead,
                "glucose_projected": round(clamped_projected, 2),
                "glucose_lower": round(g_lower, 2),
                "glucose_upper": round(g_upper, 2),
            })

        return trajectory
