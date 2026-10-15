"""
src/data/ingestion/synthea_parser.py

Synthea FHIR R4 JSON parser extracting 23 static clinical/demographic features
for Type 2 Diabetes patients, with missing data imputation and clinical quality flagging.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)

# Exact 23 EHR Feature Column Names matching GlucoTwin specification
EHR_FEATURE_COLUMNS: List[str] = [
    # Demographics (5)
    "age",
    "gender_encoded",
    "bmi",
    "bmi_category_encoded",
    "ethnicity_encoded",
    # Clinical History (6)
    "diabetes_duration_years",
    "hba1c_latest",
    "hba1c_trend",
    "fasting_glucose_avg",
    "egfr",
    "bp_sys",
    # Medications (5)
    "on_metformin",
    "on_insulin",
    "insulin_dose_units",
    "on_sulphonylurea",
    "on_sglt2",
    # Comorbidities (4)
    "has_hypertension",
    "has_dyslipidaemia",
    "has_neuropathy",
    "has_retinopathy",
    # Lifestyle (3)
    "age_group_encoded",
    "physical_activity_encoded",
    "dietary_compliance_score",
]

assert len(EHR_FEATURE_COLUMNS) == 23


class FHIRMapper:
    """Standardizes LOINC, SNOMED CT, and RxNorm terminology codes to GlucoTwin features."""

    # SNOMED CT Condition Codes
    SNOMED_T2D = {"44054006", "73211009"}       # Type 2 Diabetes Mellitus
    SNOMED_HTN = {"59621000", "38341003"}       # Essential Hypertension
    SNOMED_DYSLIPID = {"55822004", "13644009"}  # Hyperlipidemia / Dyslipidaemia
    SNOMED_NEUROPATHY = {"230572002"}            # Diabetic peripheral neuropathy
    SNOMED_RETINOPATHY = {"4855003"}             # Diabetic retinopathy

    # LOINC Observation Codes
    LOINC_HBA1C = {"4548-4", "17856-6"}          # Hemoglobin A1c
    LOINC_FASTING_GLUCOSE = {"1558-6", "1547-9"} # Fasting blood glucose
    LOINC_EGFR = {"33914-3", "62238-1"}          # Glomerular filtration rate
    LOINC_BP_SYSTOLIC = {"55284-4", "8480-6"}    # Systolic blood pressure
    LOINC_BMI = {"39156-5"}                      # Body mass index

    # RxNorm Medication Codes
    RXNORM_METFORMIN = {"860974", "6809", "861004"}
    RXNORM_INSULIN = {
        "311040",  # Insulin Glargine
        "284399",  # Insulin Lispro
        "139825",  # Regular Insulin
        "51428",   # Generic Insulin
        "847207",  # Insulin Degludec
        "106892",  # Insulin Aspart
        "847215",  # Insulin Detemir
        "261551",  # NPH Insulin
        "86009",   # Insulin Glargine recombinant
    }
    RXNORM_SULPHONYLUREA = {"310489", "4815", "4821"}     # Glipizide, Glimepiride
    RXNORM_SGLT2 = {"2200653", "1488564", "1373458"}      # Empagliflozin, Dapagliflozin


class SyntheaParser:
    """
    Parses Synthea FHIR R4 JSON outputs for Type 2 Diabetes patients.
    Supports both individual JSON directories and single FHIR Bundle files.
    """

    # Default median imputation values for Indian T2D population
    DEFAULT_MEDIANS: Dict[str, Any] = {
        "age": 52,
        "gender_encoded": 1,
        "bmi": 25.8,
        "bmi_category_encoded": 2,
        "ethnicity_encoded": 1,
        "diabetes_duration_years": 7,
        "hba1c_latest": 8.2,
        "hba1c_trend": 0.0,
        "fasting_glucose_avg": 165.0,
        "egfr": 85.0,
        "bp_sys": 132.0,
        "on_metformin": False,
        "on_insulin": False,
        "insulin_dose_units": 0.0,
        "on_sulphonylurea": False,
        "on_sglt2": False,
        "has_hypertension": False,
        "has_dyslipidaemia": False,
        "has_neuropathy": False,
        "has_retinopathy": False,
        "age_group_encoded": 1,
        "physical_activity_encoded": 1,
        "dietary_compliance_score": 0.70,
    }

    def __init__(self, reference_year: int = 2026):
        self.reference_year = reference_year

    def parse_patient(self, patient_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parses a single patient record directory or bundle JSON file.

        Args:
            patient_path: Path to patient directory containing Patient.json, etc.
                          or path to a single patient FHIR Bundle .json file.

        Returns:
            Dictionary containing 'patient_id', 23 clinical features,
            'missing_count', 'missing_ratio', and 'quality_status'.
        """
        path = Path(patient_path)
        if not path.exists():
            raise FileNotFoundError(f"Synthea patient path not found: {path}")

        patient_resource = None
        conditions = []
        observations = []
        medications = []
        patient_id = path.stem

        if path.is_file():
            # Bundle JSON
            with open(path, "r", encoding="utf-8") as f:
                bundle = json.load(f)
            entries = bundle.get("entry", [])
            for e in entries:
                res = e.get("resource", {})
                rtype = res.get("resourceType")
                if rtype == "Patient":
                    patient_resource = res
                    patient_id = res.get("id", patient_id)
                elif rtype == "Condition":
                    conditions.append(res)
                elif rtype == "Observation":
                    observations.append(res)
                elif rtype in ("MedicationRequest", "MedicationStatement"):
                    medications.append(res)
        else:
            # Directory with separated files
            patient_id = path.name
            p_file = path / "Patient.json"
            if p_file.exists():
                with open(p_file, "r", encoding="utf-8") as f:
                    patient_resource = json.load(f)
            c_file = path / "Condition.json"
            if c_file.exists():
                with open(c_file, "r", encoding="utf-8") as f:
                    conditions = json.load(f)
                    if isinstance(conditions, dict):
                        conditions = [conditions]
            o_file = path / "Observation.json"
            if o_file.exists():
                with open(o_file, "r", encoding="utf-8") as f:
                    observations = json.load(f)
                    if isinstance(observations, dict):
                        observations = [observations]
            m_file = path / "MedicationRequest.json"
            if m_file.exists():
                with open(m_file, "r", encoding="utf-8") as f:
                    medications = json.load(f)
                    if isinstance(medications, dict):
                        medications = [medications]

        # Extract features
        extracted = {}
        extracted.update(self._extract_demographics(patient_resource))
        extracted.update(self._extract_conditions(conditions))
        extracted.update(self._extract_observations(observations))
        extracted.update(self._extract_medications(medications, patient_context=extracted))

        # Check missingness before imputation
        missing_count = sum(1 for col in EHR_FEATURE_COLUMNS if col not in extracted or extracted[col] is None)
        missing_ratio = missing_count / len(EHR_FEATURE_COLUMNS)
        quality_status = "LOW_QUALITY" if missing_ratio > 0.30 else "GOOD"

        # Impute missing values with default medians
        for col in EHR_FEATURE_COLUMNS:
            if col not in extracted or extracted[col] is None:
                extracted[col] = self.DEFAULT_MEDIANS[col]

        extracted["patient_id"] = patient_id
        extracted["missing_count"] = missing_count
        extracted["missing_ratio"] = round(missing_ratio, 3)
        extracted["quality_status"] = quality_status

        return extracted

    def parse_all(
        self,
        synthea_dir: Union[str, Path],
        impute: bool = True,
        quality_threshold: float = 0.30,
    ) -> pd.DataFrame:
        """
        Parses all patient records in a Synthea root directory.
        """
        s_path = Path(synthea_dir)
        if not s_path.exists():
            raise FileNotFoundError(f"Synthea directory not found: {s_path}")

        records = []
        # Support either directories or .json bundle files
        subpaths = [p for p in s_path.iterdir() if p.is_dir() or p.suffix == ".json"]
        for p in subpaths:
            try:
                rec = self.parse_patient(p)
                records.append(rec)
            except Exception as e:
                logger.warning(f"Failed parsing Synthea record at {p}: {e}")

        if not records:
            return pd.DataFrame(columns=["patient_id"] + EHR_FEATURE_COLUMNS + ["quality_status"])

        df = pd.DataFrame(records)
        if impute:
            df = self.impute_missing_values(df)
        df = self.flag_data_quality(df, threshold=quality_threshold)
        return df

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies median imputation on numeric features and False on boolean columns.
        """
        df_out = df.copy()
        for col in EHR_FEATURE_COLUMNS:
            if col not in df_out.columns:
                df_out[col] = self.DEFAULT_MEDIANS[col]
                continue

            if isinstance(self.DEFAULT_MEDIANS[col], bool):
                df_out[col] = df_out[col].fillna(False).astype(bool)
            elif isinstance(self.DEFAULT_MEDIANS[col], (int, float)):
                med = df_out[col].median()
                if pd.isna(med):
                    med = self.DEFAULT_MEDIANS[col]
                df_out[col] = df_out[col].fillna(med)

        return df_out

    def flag_data_quality(
        self, df: pd.DataFrame, threshold: float = 0.30
    ) -> pd.DataFrame:
        """
        Calculates missingness across the 23 clinical columns.
        Sets 'quality_status' to 'LOW_QUALITY' if missing_ratio > threshold else 'GOOD'.
        """
        df_out = df.copy()
        if "missing_ratio" not in df_out.columns:
            missing_counts = df_out[EHR_FEATURE_COLUMNS].isna().sum(axis=1)
            df_out["missing_ratio"] = missing_counts / len(EHR_FEATURE_COLUMNS)

        df_out["quality_status"] = np.where(
            df_out["missing_ratio"] > threshold, "LOW_QUALITY", "GOOD"
        )
        return df_out

    def _extract_demographics(self, patient_res: Optional[dict]) -> dict:
        if not patient_res:
            return {}

        out = {}
        # Birth date -> Age
        bdate_str = patient_res.get("birthDate")
        if bdate_str:
            try:
                b_year = int(bdate_str.split("-")[0])
                age = max(18, min(95, self.reference_year - b_year))
                out["age"] = age
                out["age_group_encoded"] = self._encode_age_group(age)
            except Exception:
                pass

        # Gender
        gender = str(patient_res.get("gender", "")).lower()
        out["gender_encoded"] = 1 if gender == "male" else 0

        # Ethnicity (South Asian default = 1)
        out["ethnicity_encoded"] = 1
        return out

    def _extract_conditions(self, conditions: List[dict]) -> dict:
        out = {
            "diabetes_duration_years": None,
            "has_hypertension": False,
            "has_dyslipidaemia": False,
            "has_neuropathy": False,
            "has_retinopathy": False,
        }
        earliest_t2d_year = None

        for cond in conditions:
            codings = cond.get("code", {}).get("coding", [])
            for c in codings:
                code = str(c.get("code", ""))
                # T2D onset
                if code in FHIRMapper.SNOMED_T2D:
                    onset = cond.get("onsetDateTime", cond.get("recordedDate"))
                    if onset:
                        try:
                            year = int(str(onset).split("-")[0])
                            if earliest_t2d_year is None or year < earliest_t2d_year:
                                earliest_t2d_year = year
                        except Exception:
                            pass
                elif code in FHIRMapper.SNOMED_HTN:
                    out["has_hypertension"] = True
                elif code in FHIRMapper.SNOMED_DYSLIPID:
                    out["has_dyslipidaemia"] = True
                elif code in FHIRMapper.SNOMED_NEUROPATHY:
                    out["has_neuropathy"] = True
                elif code in FHIRMapper.SNOMED_RETINOPATHY:
                    out["has_retinopathy"] = True

        if earliest_t2d_year:
            dur = max(0, min(50, self.reference_year - earliest_t2d_year))
            out["diabetes_duration_years"] = dur

        return out

    def _extract_observations(self, observations: List[dict]) -> dict:
        out = {}
        hba1c_vals = []
        fbg_vals = []
        bp_vals = []
        bmi_vals = []
        egfr_vals = []

        for obs in observations:
            codings = obs.get("code", {}).get("coding", [])
            val_quantity = obs.get("valueQuantity", {})
            val = val_quantity.get("value")
            if val is None:
                continue

            try:
                fval = float(val)
            except Exception:
                continue

            for c in codings:
                code = str(c.get("code", ""))
                if code in FHIRMapper.LOINC_HBA1C:
                    hba1c_vals.append(fval)
                elif code in FHIRMapper.LOINC_FASTING_GLUCOSE:
                    fbg_vals.append(fval)
                elif code in FHIRMapper.LOINC_BP_SYSTOLIC:
                    bp_vals.append(fval)
                elif code in FHIRMapper.LOINC_BMI:
                    bmi_vals.append(fval)
                elif code in FHIRMapper.LOINC_EGFR:
                    egfr_vals.append(fval)

        if hba1c_vals:
            out["hba1c_latest"] = round(hba1c_vals[-1], 1)
            out["hba1c_trend"] = round(hba1c_vals[-1] - hba1c_vals[0], 2) if len(hba1c_vals) > 1 else 0.0

        if fbg_vals:
            out["fasting_glucose_avg"] = round(float(np.mean(fbg_vals)), 1)

        if bp_vals:
            out["bp_sys"] = round(float(np.mean(bp_vals)), 1)

        if bmi_vals:
            bmi = round(bmi_vals[-1], 1)
            out["bmi"] = bmi
            out["bmi_category_encoded"] = self._encode_indian_bmi_category(bmi)

        if egfr_vals:
            out["egfr"] = round(egfr_vals[-1], 1)

        return out

    def _parse_insulin_dose(self, med: dict, patient_context: Optional[dict] = None) -> float:
        """
        Extracts total daily insulin dose in units from FHIR MedicationRequest/Statement.
        Parses structured dosageInstruction (doseAndRate, doseQuantity, doseRange, timing)
        or dosage instruction text. Falls back to clinical bodyweight/BMI-based estimation
        if dosageInstruction is omitted or non-numeric.
        """
        dose_candidates: List[float] = []

        # 1. Parse structured dosageInstruction
        dosage_instructions = med.get("dosageInstruction", [])
        if isinstance(dosage_instructions, dict):
            dosage_instructions = [dosage_instructions]

        for di in dosage_instructions:
            if not isinstance(di, dict):
                continue

            # Calculate daily frequency factor from timing
            daily_factor = 1.0
            timing = di.get("timing", {})
            if isinstance(timing, dict):
                repeat = timing.get("repeat", {})
                if isinstance(repeat, dict):
                    frequency = float(repeat.get("frequency", 1))
                    period = float(repeat.get("period", 1))
                    period_unit = str(repeat.get("periodUnit", "d")).lower()
                    if period_unit in ("d", "day", "days"):
                        daily_factor = frequency / max(period, 1.0)
                    elif period_unit in ("h", "hour", "hours"):
                        daily_factor = (24.0 / max(period, 1.0)) * frequency
                    elif period_unit in ("wk", "week", "weeks"):
                        daily_factor = frequency / (7.0 * max(period, 1.0))

            # Check doseAndRate
            dar_list = di.get("doseAndRate", [])
            if isinstance(dar_list, dict):
                dar_list = [dar_list]
            for dar in dar_list:
                if not isinstance(dar, dict):
                    continue
                # doseQuantity
                dq = dar.get("doseQuantity", {})
                if isinstance(dq, dict) and "value" in dq:
                    try:
                        v = float(dq["value"])
                        if v > 0:
                            dose_candidates.append(v * daily_factor)
                    except (ValueError, TypeError):
                        pass
                # doseRange
                dr = dar.get("doseRange", {})
                if isinstance(dr, dict):
                    low = dr.get("low", {}).get("value")
                    high = dr.get("high", {}).get("value")
                    if low is not None and high is not None:
                        try:
                            avg_v = (float(low) + float(high)) / 2.0
                            if avg_v > 0:
                                dose_candidates.append(avg_v * daily_factor)
                        except (ValueError, TypeError):
                            pass

            # Check direct doseQuantity on dosageInstruction
            dq_direct = di.get("doseQuantity", {})
            if isinstance(dq_direct, dict) and "value" in dq_direct:
                try:
                    v = float(dq_direct["value"])
                    if v > 0:
                        dose_candidates.append(v * daily_factor)
                except (ValueError, TypeError):
                    pass

            # Check textual dosage instructions (e.g., "Take 32 units once daily")
            di_text = di.get("text", "")
            if di_text:
                match = re.search(r"(\d+(?:\.\d+)?)\s*(?:units?|u\b|iu\b)", str(di_text), re.IGNORECASE)
                if match:
                    try:
                        v = float(match.group(1))
                        if "twice" in str(di_text).lower() or "bid" in str(di_text).lower():
                            v *= 2.0
                        elif "three" in str(di_text).lower() or "tid" in str(di_text).lower():
                            v *= 3.0
                        if v > 0:
                            dose_candidates.append(v)
                    except (ValueError, TypeError):
                        pass

        # Also check top-level text if available
        med_text = med.get("text", {}).get("div", "") if isinstance(med.get("text"), dict) else ""
        if med_text and not dose_candidates:
            match = re.search(r"(\d+(?:\.\d+)?)\s*(?:units?|u\b|iu\b)", str(med_text), re.IGNORECASE)
            if match:
                try:
                    v = float(match.group(1))
                    if v > 0:
                        dose_candidates.append(v)
                except (ValueError, TypeError):
                    pass

        # If a structured or textual dose was successfully extracted from FHIR:
        if dose_candidates:
            total_dose = sum(dose_candidates) if len(dose_candidates) <= 2 else max(dose_candidates)
            return round(float(np.clip(total_dose, 5.0, 150.0)), 1)

        # 2. Dynamic Clinical Estimation Fallback:
        # For Type 2 Diabetes, guideline basal/total daily dose is 0.40 - 0.55 U/kg.
        # Estimate patient bodyweight from BMI (assuming adult mean height 1.65m): weight = bmi * (1.65)^2
        patient_context = patient_context or {}
        bmi = patient_context.get("bmi")
        if bmi is not None and bmi > 0:
            est_weight_kg = float(bmi) * (1.65 ** 2)
            est_dose = est_weight_kg * 0.45
            return round(float(np.clip(est_dose, 15.0, 80.0)), 1)

        # Conservative fallback if no BMI or dosage data exists
        return 32.0

    def _extract_medications(self, medications: List[dict], patient_context: Optional[dict] = None) -> dict:
        out = {
            "on_metformin": False,
            "on_insulin": False,
            "insulin_dose_units": 0.0,
            "on_sulphonylurea": False,
            "on_sglt2": False,
        }

        insulin_doses: List[float] = []

        for med in medications:
            codings = med.get("medicationCodeableConcept", {}).get("coding", [])
            for c in codings:
                code = str(c.get("code", ""))
                if code in FHIRMapper.RXNORM_METFORMIN:
                    out["on_metformin"] = True
                elif code in FHIRMapper.RXNORM_INSULIN:
                    out["on_insulin"] = True
                    dose = self._parse_insulin_dose(med, patient_context)
                    if dose > 0:
                        insulin_doses.append(dose)
                elif code in FHIRMapper.RXNORM_SULPHONYLUREA:
                    out["on_sulphonylurea"] = True
                elif code in FHIRMapper.RXNORM_SGLT2:
                    out["on_sglt2"] = True

        if out["on_insulin"]:
            if insulin_doses:
                out["insulin_dose_units"] = round(float(np.mean(insulin_doses)), 1)
            else:
                out["insulin_dose_units"] = self._parse_insulin_dose({}, patient_context)
        else:
            out["insulin_dose_units"] = 0.0

        return out

    def _encode_indian_bmi_category(self, bmi: Optional[float]) -> int:
        """Indian BMI cutoffs: <18.5: 0, 18.5-22.9: 1, 23.0-27.4: 2, >=27.5: 3."""
        if bmi is None:
            return 1
        if bmi < 18.5:
            return 0
        if bmi <= 22.9:
            return 1
        if bmi <= 27.4:
            return 2
        return 3

    def _encode_age_group(self, age: Optional[int]) -> int:
        """Age groups: 35-45: 0, 46-55: 1, 56-65: 2, 66+: 3."""
        if age is None or age < 46:
            return 0
        if age <= 55:
            return 1
        if age <= 65:
            return 2
        return 3
