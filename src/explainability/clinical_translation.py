"""
src/explainability/clinical_translation.py
Doctor-Facing Clinical Translation Dictionary for Engineered Feature Names.
Translates mathematical and time-series feature names into clear clinical language.
"""

from typing import Any, Dict

CLINICAL_FEATURE_MAPPING: Dict[str, str] = {
    "glucose_current": "Current CGM blood glucose level",
    "glucose_roc_current": "Glucose rate of change velocity",
    "glucose_mean_1h": "1-hour rolling average glucose",
    "glucose_std_1h": "1-hour glycemic variability (standard deviation)",
    "glucose_min_1h": "1-hour nadir glucose",
    "glucose_max_1h": "1-hour peak glucose",
    "glucose_mean_4h": "4-hour sustained glucose level",
    "glucose_std_4h": "4-hour glycemic volatility",
    "tir_24h": "Time in Target Range (70-180 mg/dL)",
    "tar_24h": "Time above range (>180 mg/dL)",
    "tbr_24h": "Time below range (<70 mg/dL)",
    "mage_approx": "Mean Amplitude of Glycemic Excursions (MAGE)",
    "hrv_rmssd_ms": "Heart rate variability (RMSSD parasympathetic marker)",
    "hrv_sdnn_ms": "Overall autonomic autonomic regulation (SDNN)",
    "hrv_suppression_flag": "Autonomic HRV suppression indicator",
    "hba1c_latest": "Latest HbA1c laboratory baseline",
    "egfr": "Estimated Glomerular Filtration Rate (Renal function)",
    "bp_sys": "Systolic blood pressure",
    "bmi": "Body Mass Index",
    "diabetes_duration_years": "Duration of Type 2 Diabetes diagnosis",
    "time_since_last_meal_min": "Time since last meal consumption",
    "current_iob": "Active insulin-on-board (pharmacokinetic residue)",
    "insulin_dose_units": "Prescribed insulin dosage",
}


def translate_feature(feature_name: str, value: Any = None) -> str:
    """
    Convert technical feature identifier into clinician-comprehensible medical language.
    Includes context values where helpful.
    """
    feat_lower = feature_name.lower()
    
    # Direct dictionary lookup
    if feat_lower in CLINICAL_FEATURE_MAPPING:
        label = CLINICAL_FEATURE_MAPPING[feat_lower]
        if value is not None and isinstance(value, (int, float)):
            return f"{label} ({value:.2f})"
        return label

    # Pattern-matched heuristic mappings
    if "roc" in feat_lower:
        return f"Glucose rate of change ({value})" if value is not None else "Glucose rate of change"
    if "hrv" in feat_lower and "suppression" in feat_lower:
        return "Autonomic HRV suppression marker"
    if "hba1c" in feat_lower:
        return f"Latest HbA1c ({value}%)" if value is not None else "Latest HbA1c"
    if "meal" in feat_lower:
        return f"Time since last meal ({value} min)" if value is not None else "Time since last meal"
    if "tir" in feat_lower:
        return "Time-in-Range (TIR)"
    if "insulin" in feat_lower:
        return "Insulin therapy dynamic"
    if "glucose" in feat_lower:
        return f"CGM glucose marker ({feature_name})"

    # Clean fallback format
    formatted = feature_name.replace("_", " ").title()
    return f"{formatted} ({value})" if value is not None else formatted
