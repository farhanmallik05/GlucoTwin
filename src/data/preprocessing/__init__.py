"""
Preprocessing package: Clinical validator, clinical indices, 89-feature engineering, and adverse event labeler.
"""

from src.data.preprocessing.clinical_validator import ClinicalValidator, ValidationResult, ValidationReport
from src.data.preprocessing.clinical_indices import (
    ClinicalIndicesCalculator,
    calculate_tir,
    calculate_tar,
    calculate_tbr,
    calculate_gmi,
    calculate_cv,
    compute_clinical_indices,
)
from src.data.preprocessing.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.data.preprocessing.labeler import label_adverse_events

__all__ = [
    "ClinicalValidator",
    "ValidationResult",
    "ValidationReport",
    "ClinicalIndicesCalculator",
    "calculate_tir",
    "calculate_tar",
    "calculate_tbr",
    "calculate_gmi",
    "calculate_cv",
    "compute_clinical_indices",
    "FeatureEngineer",
    "FEATURE_COLUMNS",
    "label_adverse_events",
]
