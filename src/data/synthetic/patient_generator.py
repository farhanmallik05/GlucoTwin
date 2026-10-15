"""
src/data/synthetic/patient_generator.py

Synthetic Indian Type 2 Diabetes EHR profile generator across general population
and 4 critical clinical edge-case cohorts (Lean T2D, Elderly Brittle, Insulin-Naive, Tight Control),
matching ICMR-INDIAB epidemiological parameters and 100% compliant with ClinicalValidator.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class PatientGenerator:
    """
    Generates synthetic Indian T2D patients matching ICMR-INDIAB epidemiological parameters:
    - 23 standardized clinical/demographic features
    - Full compliance with ClinicalValidator ranges and contraindication rules
    """

    def __init__(self, seed: Optional[int] = None, random_seed: Optional[int] = None):
        actual_seed = seed if seed is not None else (random_seed if random_seed is not None else 42)
        self.rng = np.random.default_rng(actual_seed)

    def generate_patient(self, patient_id: str, cohort: str = "general") -> Dict[str, Any]:
        """Generates a single patient EHR record for the specified cohort."""
        if cohort == "lean_t2d":
            return self._generate_lean_t2d(patient_id)
        elif cohort == "elderly_brittle":
            return self._generate_elderly_brittle(patient_id)
        elif cohort == "insulin_naive_high_hba1c":
            return self._generate_insulin_naive(patient_id)
        elif cohort == "tight_control_insulin":
            return self._generate_tight_control(patient_id)
        else:
            return self._generate_general_patient(patient_id)

    def _generate_general_patient(self, patient_id: str) -> Dict[str, Any]:
        # Age distribution: 35-45 (30%), 45-55 (35%), 55-65 (25%), 65+ (10%)
        age_group = int(self.rng.choice([0, 1, 2, 3], p=[0.30, 0.35, 0.25, 0.10]))
        age_ranges = [(35, 45), (46, 55), (56, 65), (66, 80)]
        age = int(self.rng.integers(age_ranges[age_group][0], age_ranges[age_group][1] + 1))
        gender_encoded = int(self.rng.choice([0, 1], p=[0.48, 0.52]))

        # BMI distribution: 19-22.9 (20%), 23-26.9 (40%), 27-29.9 (25%), 30-38 (15%)
        bmi_cat = int(self.rng.choice([1, 2, 3], p=[0.25, 0.45, 0.30]))
        bmi_ranges = {
            1: (19.0, 22.9),
            2: (23.0, 27.4),
            3: (27.5, 36.0),
        }
        bmi = float(np.round(self.rng.uniform(*bmi_ranges[bmi_cat]), 1))

        # HbA1c distribution: 6.5 - 12.0
        hba1c_cat = int(self.rng.choice([0, 1, 2, 3, 4], p=[0.20, 0.25, 0.30, 0.15, 0.10]))
        hba1c_ranges = [(6.2, 6.9), (7.0, 7.9), (8.0, 8.9), (9.0, 9.9), (10.0, 12.5)]
        hba1c = float(np.round(self.rng.uniform(*hba1c_ranges[hba1c_cat]), 1))
        hba1c_trend = float(np.round(self.rng.normal(0.0, 0.35), 2))

        # Fasting glucose correlated with HbA1c
        fasting_glucose = float(np.round(28.7 * hba1c - 46.7 + self.rng.normal(0, 12), 1))
        fasting_glucose = float(np.clip(fasting_glucose, 80.0, 350.0))

        # Diabetes duration
        max_duration = min(30, max(1, age - 28))
        duration = int(self.rng.integers(1, max_duration + 1))

        # eGFR and Blood Pressure
        egfr = float(np.round(max(25.0, 110.0 - (age * 0.55) + self.rng.normal(0, 8)), 1))
        bp_sys = float(np.round(115.0 + (age * 0.32) + self.rng.normal(0, 8), 1))
        bp_sys = float(np.clip(bp_sys, 95.0, 190.0))

        # Medications
        on_insulin = bool(self.rng.random() < 0.45)
        insulin_dose = float(np.round(self.rng.uniform(20.0, 60.0), 1)) if on_insulin else 0.0

        if egfr < 30.0:
            on_metformin = False
        else:
            on_metformin = bool(self.rng.random() < 0.82)

        on_sulphonylurea = bool(self.rng.random() < 0.28)
        on_sglt2 = bool(self.rng.random() < 0.30 and egfr >= 30.0)

        # Comorbidities
        has_htn = bool(self.rng.random() < 0.70 or bp_sys > 140.0)
        has_dyslip = bool(self.rng.random() < 0.60)
        has_neuro = bool(self.rng.random() < (0.12 + duration * 0.015))
        has_retino = bool(self.rng.random() < (0.08 + duration * 0.012))

        # Lifestyle
        phys_act = int(self.rng.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.20, 0.10]))
        diet_score = float(np.round(np.clip(0.68 - (hba1c - 7.0) * 0.04 + self.rng.normal(0, 0.08), 0.20, 0.95), 2))

        return {
            "patient_id": patient_id,
            "age": age,
            "gender_encoded": gender_encoded,
            "bmi": bmi,
            "bmi_category_encoded": bmi_cat,
            "ethnicity_encoded": 1,  # South Asian
            "diabetes_duration_years": duration,
            "hba1c_latest": hba1c,
            "hba1c_trend": hba1c_trend,
            "fasting_glucose_avg": fasting_glucose,
            "egfr": egfr,
            "bp_sys": bp_sys,
            "on_metformin": on_metformin,
            "on_insulin": on_insulin,
            "insulin_dose_units": insulin_dose,
            "on_sulphonylurea": on_sulphonylurea,
            "on_sglt2": on_sglt2,
            "has_hypertension": has_htn,
            "has_dyslipidaemia": has_dyslip,
            "has_neuropathy": has_neuro,
            "has_retinopathy": has_retino,
            "age_group_encoded": age_group,
            "physical_activity_encoded": phys_act,
            "dietary_compliance_score": diet_score,
            "cohort": "general",
        }

    def _generate_lean_t2d(self, patient_id: str) -> Dict[str, Any]:
        """Cohort 1: Lean T2D (BMI 18.5-22.9, high HbA1c 8-10%, younger onset)."""
        p = self._generate_general_patient(patient_id)
        p["cohort"] = "lean_t2d"
        p["age"] = int(self.rng.integers(35, 48))
        p["age_group_encoded"] = 0
        p["bmi"] = float(np.round(self.rng.uniform(19.0, 22.8), 1))
        p["bmi_category_encoded"] = 1
        p["hba1c_latest"] = float(np.round(self.rng.uniform(8.2, 10.2), 1))
        p["fasting_glucose_avg"] = float(np.round(28.7 * p["hba1c_latest"] - 46.7 + self.rng.normal(0, 10), 1))
        p["on_sulphonylurea"] = True
        return p

    def _generate_elderly_brittle(self, patient_id: str) -> Dict[str, Any]:
        """Cohort 2: Elderly Brittle (age 70+, long duration, CKD eGFR < 60, high hypo risk)."""
        p = self._generate_general_patient(patient_id)
        p["cohort"] = "elderly_brittle"
        p["age"] = int(self.rng.integers(71, 86))
        p["age_group_encoded"] = 3
        p["diabetes_duration_years"] = int(self.rng.integers(20, 36))
        p["on_insulin"] = True
        p["insulin_dose_units"] = float(np.round(self.rng.uniform(30.0, 65.0), 1))
        p["egfr"] = float(np.round(self.rng.uniform(28.0, 52.0), 1))
        if p["egfr"] < 30.0:
            p["on_metformin"] = False
        p["has_hypertension"] = True
        p["has_dyslipidaemia"] = True
        p["has_neuropathy"] = True
        p["has_retinopathy"] = bool(self.rng.random() < 0.70)
        return p

    def _generate_insulin_naive(self, patient_id: str) -> Dict[str, Any]:
        """Cohort 3: Insulin-Naive High HbA1c (>9%, oral only, high hyper risk)."""
        p = self._generate_general_patient(patient_id)
        p["cohort"] = "insulin_naive_high_hba1c"
        p["hba1c_latest"] = float(np.round(self.rng.uniform(9.4, 13.0), 1))
        p["fasting_glucose_avg"] = float(np.round(28.7 * p["hba1c_latest"] - 46.7 + self.rng.normal(10, 15), 1))
        p["on_insulin"] = False
        p["insulin_dose_units"] = 0.0
        p["on_metformin"] = True if p["egfr"] >= 30.0 else False
        p["on_sulphonylurea"] = True
        p["dietary_compliance_score"] = float(np.round(self.rng.uniform(0.15, 0.38), 2))
        return p

    def _generate_tight_control(self, patient_id: str) -> Dict[str, Any]:
        """Cohort 4: Tight Control on Insulin (HbA1c 6.4-7.0%, active, high hypo risk)."""
        p = self._generate_general_patient(patient_id)
        p["cohort"] = "tight_control_insulin"
        p["hba1c_latest"] = float(np.round(self.rng.uniform(6.4, 7.0), 1))
        p["fasting_glucose_avg"] = float(np.round(self.rng.uniform(85.0, 115.0), 1))
        p["on_insulin"] = True
        p["insulin_dose_units"] = float(np.round(self.rng.uniform(22.0, 48.0), 1))
        p["physical_activity_encoded"] = int(self.rng.choice([2, 3]))
        p["dietary_compliance_score"] = float(np.round(self.rng.uniform(0.78, 0.95), 2))
        return p

    def generate_population(
        self,
        n_patients: int = 1000,
        include_edge_cases: bool = True,
    ) -> pd.DataFrame:
        """
        Generates population of patients. If include_edge_cases is True and n_patients >= 200,
        includes 50 patients of each of the 4 clinical edge cohorts.
        """
        patients = []
        if include_edge_cases and n_patients >= 200:
            for i in range(50):
                patients.append(self.generate_patient(f"PAT-LEAN-{i+1:03d}", "lean_t2d"))
            for i in range(50):
                patients.append(self.generate_patient(f"PAT-BRIT-{i+1:03d}", "elderly_brittle"))
            for i in range(50):
                patients.append(self.generate_patient(f"PAT-NAIV-{i+1:03d}", "insulin_naive_high_hba1c"))
            for i in range(50):
                patients.append(self.generate_patient(f"PAT-TIGH-{i+1:03d}", "tight_control_insulin"))

            remaining = n_patients - 200
            for i in range(remaining):
                patients.append(self.generate_patient(f"PAT-GEN-{i+1:04d}", "general"))
        else:
            for i in range(n_patients):
                patients.append(self.generate_patient(f"PAT-GEN-{i+1:04d}", "general"))

        return pd.DataFrame(patients)


class EdgeCaseGenerator:
    """Convenience wrapper for generating the 200 edge-case patient cohort."""

    def __init__(self, seed: Optional[int] = None, random_seed: Optional[int] = None):
        actual_seed = seed if seed is not None else (random_seed if random_seed is not None else 42)
        self.generator = PatientGenerator(seed=actual_seed)

    def generate_edge_cases(self, count_per_cohort: int = 50) -> pd.DataFrame:
        """Generates count_per_cohort patients for each edge cohort."""
        records = []
        cohorts = [
            ("PAT-LEAN-", "lean_t2d"),
            ("PAT-BRIT-", "elderly_brittle"),
            ("PAT-NAIV-", "insulin_naive_high_hba1c"),
            ("PAT-TIGH-", "tight_control_insulin"),
        ]
        for prefix, cohort in cohorts:
            for i in range(count_per_cohort):
                pid = f"{prefix}{i+1:03d}"
                records.append(self.generator.generate_patient(pid, cohort=cohort))
        return pd.DataFrame(records)
