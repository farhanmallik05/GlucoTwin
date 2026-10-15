"""
src/data/preprocessing/clinical_validator.py

Comprehensive EHR and clinical profile validator adhering to physiological
reference ranges, ICMR-INDIAB epidemiological parameters, and medical contraindication rules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Captures patient-level clinical validation outcome."""
    patient_id: Optional[str] = None
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class ValidationReport:
    """Aggregate cohort validation report."""
    total_patients: int
    valid_patients: int
    patients_with_warnings: int
    patients_with_errors: int
    field_violations: Dict[str, int]
    rule_violations: Dict[str, int]
    cleaned_df: pd.DataFrame
    error_count: int = 0
    valid_percentage: float = 100.0


# Authoritative clinical reference bounds
CLINICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "age": (18.0, 95.0),
    "bmi": (15.0, 55.0),
    "hba1c_latest": (4.0, 15.0),
    "fasting_glucose_avg": (40.0, 600.0),
    "egfr": (5.0, 120.0),
    "bp_sys": (80.0, 220.0),
    "insulin_dose_units": (0.0, 200.0),
    "diabetes_duration_years": (0.0, 50.0),
    "dietary_compliance_score": (0.0, 1.0),
}

FIELD_ALIASES: Dict[str, str] = {
    "blood_pressure_sys": "bp_sys",
    "bp_systolic": "bp_sys",
    "hba1c": "hba1c_latest",
    "fasting_glucose": "fasting_glucose_avg",
    "insulin_dose": "insulin_dose_units",
    "diabetes_duration": "diabetes_duration_years",
}


class ClinicalValidator:
    """
    Validates clinical profiles against physiological bounds and clinical contraindications.
    """

    def __init__(self, bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.bounds = bounds or CLINICAL_BOUNDS

    def validate(self, patient: Union[Dict[str, Any], pd.Series]) -> ValidationResult:
        """Alias for validate_patient for backward and test suite compatibility."""
        return self.validate_patient(patient)

    def validate_patient(self, patient: Union[Dict[str, Any], pd.Series]) -> ValidationResult:
        """
        Validates a single patient record against clinical bounds and rules.

        Args:
            patient: Dictionary or Series of EHR features.

        Returns:
            ValidationResult with boolean is_valid, warnings, and errors.
        """
        if isinstance(patient, pd.Series):
            p_dict = patient.to_dict()
        else:
            p_dict = dict(patient)

        # Normalize field aliases
        for alias, standard in FIELD_ALIASES.items():
            if alias in p_dict and standard not in p_dict:
                p_dict[standard] = p_dict[alias]

        patient_id = str(p_dict.get("patient_id", "UNKNOWN"))
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Range Validation
        for field_name, (low, high) in self.bounds.items():
            if field_name in p_dict:
                val = p_dict[field_name]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    try:
                        fval = float(val)
                        if fval < low or fval > high:
                            errors.append(
                                f"FIELD_OUT_OF_BOUNDS: {field_name}={fval} outside [{low}, {high}]"
                            )
                    except (ValueError, TypeError):
                        errors.append(f"INVALID_TYPE: {field_name}={val} is not numeric")

        # 2. Rule 1: Insulin Inconsistency
        on_insulin = bool(p_dict.get("on_insulin", False))
        dose = float(p_dict.get("insulin_dose_units", 0.0) or 0.0)
        if not on_insulin and dose > 0.0:
            errors.append(
                f"INCONSISTENT: on_insulin is False but insulin_dose_units is {dose} > 0"
            )

        # 3. Rule 2: Conflicting Glycemic Markers
        hba1c = p_dict.get("hba1c_latest")
        fasting_g = p_dict.get("fasting_glucose_avg")
        if hba1c is not None and fasting_g is not None:
            try:
                h_val = float(hba1c)
                f_val = float(fasting_g)
                if h_val < 6.0 and f_val > 200.0:
                    errors.append(
                        f"INCONSISTENT: Severe conflict between HbA1c ({h_val}%) and fasting glucose ({f_val} mg/dL)"
                    )
            except (ValueError, TypeError):
                pass

        # 4. Rule 3: Metformin Contraindication in Severe CKD (eGFR < 30)
        egfr = p_dict.get("egfr")
        on_metformin = bool(p_dict.get("on_metformin", False))
        if egfr is not None:
            try:
                egfr_val = float(egfr)
                if egfr_val < 30.0 and on_metformin:
                    warnings.append(
                        f"CLINICAL WARNING: Metformin contraindicated in severe CKD (eGFR={egfr_val} < 30 mL/min/1.73m²) due to lactic acidosis risk"
                    )
            except (ValueError, TypeError):
                pass

        # 5. Rule 4: Early Onset / T1D Masquerade
        age = p_dict.get("age")
        duration = p_dict.get("diabetes_duration_years")
        if age is not None and duration is not None:
            try:
                age_val = float(age)
                dur_val = float(duration)
                if age_val < 25.0 and dur_val > 15.0:
                    warnings.append(
                        f"CLINICAL WARNING: Onset before age 10 (age={age_val}, duration={dur_val}); potential T1D masquerade"
                    )
                if dur_val > age_val:
                    errors.append(
                        f"INCONSISTENT: Diabetes duration ({dur_val}y) exceeds patient age ({age_val}y)"
                    )
            except (ValueError, TypeError):
                pass

        # 6. Rule 5: High Insulin Dose
        if dose > 150.0:
            warnings.append(
                f"CLINICAL WARNING: Extremely high insulin dose ({dose} units/day)"
            )

        # 7. Rule 6: Missingness Threshold (>30%)
        required_fields = list(self.bounds.keys())
        missing_count = sum(
            1 for k in required_fields if (k not in p_dict or p_dict[k] is None or (isinstance(p_dict[k], float) and np.isnan(p_dict[k])))
        )
        missing_ratio = missing_count / len(required_fields)
        if missing_ratio > 0.30:
            warnings.append(
                f"LOW_QUALITY: Patient has {missing_ratio:.1%} missing fields (>30% threshold)"
            )

        is_valid = len(errors) == 0
        return ValidationResult(
            patient_id=patient_id,
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
        )

    def validate_dataset(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validates all patient rows in DataFrame, returns summary metrics and clean DataFrame.
        """
        total = len(df)
        valid_indices = []
        field_violations: Dict[str, int] = {}
        rule_violations: Dict[str, int] = {}
        patients_with_warnings = 0
        patients_with_errors = 0
        total_error_count = 0

        for idx, row in df.iterrows():
            res = self.validate_patient(row)
            if res.is_valid:
                valid_indices.append(idx)
            else:
                patients_with_errors += 1
                total_error_count += len(res.errors)

            if len(res.warnings) > 0:
                patients_with_warnings += 1

            for err in res.errors:
                if "FIELD_OUT_OF_BOUNDS" in err or "INVALID_TYPE" in err:
                    col = err.split(":")[1].split("=")[0].strip()
                    field_violations[col] = field_violations.get(col, 0) + 1
                else:
                    rule_violations[err] = rule_violations.get(err, 0) + 1

        cleaned_df = df.loc[valid_indices].copy().reset_index(drop=True)
        valid_patients = len(cleaned_df)
        valid_pct = (valid_patients / total * 100.0) if total > 0 else 100.0

        return ValidationReport(
            total_patients=total,
            valid_patients=valid_patients,
            patients_with_warnings=patients_with_warnings,
            patients_with_errors=patients_with_errors,
            field_violations=field_violations,
            rule_violations=rule_violations,
            cleaned_df=cleaned_df,
            error_count=total_error_count,
            valid_percentage=valid_pct,
        )

    def generate_validation_report(self, report: Union[ValidationReport, pd.DataFrame]) -> str:
        """
        Generates a human-readable Markdown validation summary report.
        """
        if isinstance(report, pd.DataFrame):
            report = self.validate_dataset(report)

        lines = [
            "# Clinical Dataset Validation Report",
            "",
            f"- **Total Patients**: {report.total_patients}",
            f"- **Valid Patients**: {report.valid_patients} ({report.valid_percentage:.1f}%)",
            f"- **Patients with Warnings**: {report.patients_with_warnings}",
            f"- **Patients with Errors**: {report.patients_with_errors}",
            f"- **Total Errors Count**: {report.error_count}",
            "",
            "## Field Violations",
        ]
        if report.field_violations:
            for k, v in report.field_violations.items():
                lines.append(f"- `{k}`: {v} violations")
        else:
            lines.append("- None")

        lines.extend(["", "## Rule Violations"])
        if report.rule_violations:
            for k, v in report.rule_violations.items():
                lines.append(f"- {k}: {v} occurrences")
        else:
            lines.append("- None")

        return "\n".join(lines)
