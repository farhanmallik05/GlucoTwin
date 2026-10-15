"""
src/twin/iob_model.py
Biexponential Pharmacokinetic Insulin-on-Board (IOB) Model.
Based on Hovorka et al. (2004) and Mudaliar et al. (1999) insulin absorption kinetics.
"""

from typing import Dict
import numpy as np


class IOBModel:
    """
    Biexponential pharmacokinetic model for insulin-on-board calculation.
    Models the two-compartment subcutaneous absorption and clearance kinetics
    of injected insulin analogues.
    """

    INSULIN_DURATION: Dict[str, float] = {
        "rapid_acting": 4.0,    # Insulin Lispro / Aspart / Glulisine
        "short_acting": 6.0,    # Regular human insulin
        "long_acting": 24.0,    # Insulin Glargine / Degludec
        "intermediate": 16.0,   # NPH insulin
        "default": 4.0,
    }

    def __init__(self, insulin_type: str = "rapid_acting", duration_hours: float = None):
        self.insulin_type = insulin_type
        if duration_hours is not None:
            self.duration_hours = float(duration_hours)
        else:
            self.duration_hours = self.INSULIN_DURATION.get(insulin_type, 4.0)

        # Peak action occurs at approximately 35% of total duration
        self.peak_hours = self.duration_hours * 0.35

    def calculate_iob(self, dose_units: float, hours_since_injection: float) -> float:
        """
        Calculate remaining active insulin (units) using biexponential decay.
        IOB(t) = dose * [A * exp(-t / tau1) + (1 - A) * exp(-t / tau2)]
        """
        if dose_units <= 0:
            return 0.0
        if hours_since_injection <= 0:
            return float(dose_units)
        if hours_since_injection >= self.duration_hours:
            return 0.0

        t = hours_since_injection
        duration = self.duration_hours
        peak = self.peak_hours

        # Characteristic time constants
        tau1 = peak / np.log(2.0)
        tau2 = (duration - peak) / np.log(2.0)

        # Fast compartment fraction
        a_fast = 0.7
        iob_fraction = a_fast * np.exp(-t / tau1) + (1.0 - a_fast) * np.exp(-t / tau2)

        # Bound strictly between 0 and dose_units
        calculated_iob = float(dose_units * max(0.0, min(1.0, iob_fraction)))
        return calculated_iob

    def estimate_glucose_lowering(self, iob_units: float, isf: float) -> float:
        """
        Estimate projected glucose drop from currently active insulin.
        drop (mg/dL) = IOB (units) * ISF (mg/dL/unit)
        """
        if iob_units <= 0:
            return 0.0
        return float(iob_units * isf)

    def high_iob_risk(self, iob_units: float, current_glucose: float, isf: float) -> bool:
        """
        Flag whether active IOB is sufficient to drive glucose into hypoglycemia (< 80 mg/dL).
        """
        projected_drop = self.estimate_glucose_lowering(iob_units, isf)
        projected_glucose = current_glucose - projected_drop
        return projected_glucose < 80.0
