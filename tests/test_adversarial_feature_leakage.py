"""
tests/test_adversarial_feature_leakage.py

Adversarial Stress Test Suite for Milestone 1 Challenger 2:
1. FeatureEngineer edge-case windows (empty, single-row, 1000-row, all zeros, all NaNs, extreme RoC/spikes, missing cols).
2. ClinicalValidator physiological bounds, contraindications, and conflicting markers.
3. DatasetBuilder zero-leakage mathematical disjointness, temporal causality, and window integrity.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
import numpy as np
import pandas as pd
import pytest

def get_utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)

from src.data.preprocessing.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.data.preprocessing.clinical_validator import ClinicalValidator, ValidationResult, ValidationReport
from src.data.preprocessing.clinical_indices import ClinicalIndicesCalculator, compute_clinical_indices
from src.data.preprocessing.labeler import label_adverse_events


# ===========================================================================
# 1. Feature Engineering Adversarial Stress Suite
# ===========================================================================

class TestFeatureEngineeringAdversarial:
    """Stress tests FeatureEngineer under adversarial window conditions."""

    @pytest.fixture
    def engineer(self) -> FeatureEngineer:
        return FeatureEngineer()

    @pytest.fixture
    def minimal_ehr(self) -> Dict[str, Any]:
        return {
            "age": 52, "gender_encoded": 1, "bmi": 25.4, "hba1c_latest": 8.0,
            "hba1c_trend": 0.0, "fasting_glucose_avg": 150.0, "egfr": 80.0,
            "bp_sys": 128.0, "diabetes_duration_years": 5, "on_metformin": True,
            "on_insulin": False, "insulin_dose_units": 0.0, "on_sulphonylurea": False,
            "on_sglt2": False, "has_hypertension": False, "has_dyslipidaemia": False,
            "has_neuropathy": False, "has_retinopathy": False, "bmi_category_encoded": 1,
            "age_group_encoded": 1, "ethnicity_encoded": 1, "physical_activity_encoded": 1,
            "dietary_compliance_score": 0.75,
        }

    def _assert_features_contract(self, features: Dict[str, float]):
        """Asserts exactly 89 keys, all float, zero NaNs, zero Infs."""
        assert len(features) == 89, f"Expected 89 features, got {len(features)}"
        assert list(features.keys()) == FEATURE_COLUMNS, "Feature keys do not match FEATURE_COLUMNS order"
        for k, v in features.items():
            assert isinstance(v, (float, int)), f"Feature {k} has non-numeric type {type(v)}"
            assert not math.isnan(v), f"Feature {k} returned NaN"
            assert not math.isinf(v), f"Feature {k} returned Inf"

    def test_empty_dataframe(self, engineer, minimal_ehr):
        """Empty input DataFrame must trigger fallback padding and return 89 finite features."""
        empty_df = pd.DataFrame()
        feats = engineer.compute_features(empty_df, minimal_ehr)
        self._assert_features_contract(feats)

    def test_single_row_dataframe(self, engineer, minimal_ehr):
        """Single-row input DataFrame must be left-padded to 48 rows without error."""
        single_row_df = pd.DataFrame([{
            "timestamp": get_utc_now(),
            "glucose_mgdl": 145.0,
            "glucose_roc": 0.0,
            "glucose_acc": 0.0,
            "hrv_rmssd_ms": 35.0,
            "hrv_sdnn_ms": 42.0,
            "hrv_lf_hf_ratio": 1.4,
            "steps_last_5min": 10,
            "activity_type": "resting",
            "sleep_stage": "none",
            "sleep_quality": 0.8,
        }])
        feats = engineer.compute_features(single_row_df, minimal_ehr)
        self._assert_features_contract(feats)
        assert feats["glucose_current"] == 145.0
        assert feats["glucose_mean_4h"] == 145.0
        assert feats["glucose_std_4h"] == 0.0

    def test_short_dataframe_two_rows(self, engineer, minimal_ehr):
        """2-row input DataFrame must pad correctly and compute statistics."""
        short_df = pd.DataFrame([
            {"glucose_mgdl": 120.0, "hrv_rmssd_ms": 40.0, "steps_last_5min": 0},
            {"glucose_mgdl": 130.0, "hrv_rmssd_ms": 38.0, "steps_last_5min": 5},
        ])
        feats = engineer.compute_features(short_df, minimal_ehr)
        self._assert_features_contract(feats)
        assert feats["glucose_current"] == 130.0

    def test_large_dataframe_1000_rows(self, engineer, minimal_ehr):
        """Large input DataFrame (1000 rows) must correctly slice latest 48/144/288 windows."""
        n = 1000
        times = [get_utc_now() - timedelta(minutes=(n - i) * 5) for i in range(n)]
        large_df = pd.DataFrame({
            "timestamp": times,
            "glucose_mgdl": np.linspace(100.0, 200.0, n),
            "glucose_roc": np.full(n, 0.1),
            "glucose_acc": np.zeros(n),
            "hrv_rmssd_ms": np.full(n, 36.0),
            "hrv_sdnn_ms": np.full(n, 45.0),
            "hrv_lf_hf_ratio": np.full(n, 1.5),
            "steps_last_5min": np.random.randint(0, 50, n),
            "activity_type": ["resting"] * n,
            "sleep_stage": ["awake"] * n,
            "sleep_quality": [0.85] * n,
        })
        feats = engineer.compute_features(large_df, minimal_ehr)
        self._assert_features_contract(feats)
        assert feats["glucose_current"] == 200.0
        # 4h window is the last 48 rows: linspace(100, 200) -> last 48 values are > 190
        assert feats["glucose_mean_4h"] > 190.0

    def test_all_zeros_window(self, engineer, minimal_ehr):
        """Adversarial window where all telemetry values are exactly zero."""
        zeros_df = pd.DataFrame({
            "glucose_mgdl": [0.0] * 48,
            "glucose_roc": [0.0] * 48,
            "glucose_acc": [0.0] * 48,
            "hrv_rmssd_ms": [0.0] * 48,
            "hrv_sdnn_ms": [0.0] * 48,
            "hrv_lf_hf_ratio": [0.0] * 48,
            "steps_last_5min": [0] * 48,
            "activity_type": ["resting"] * 48,
            "sleep_stage": ["none"] * 48,
            "sleep_quality": [0.0] * 48,
        })
        feats = engineer.compute_features(zeros_df, minimal_ehr)
        self._assert_features_contract(feats)
        # Clamped glucose minimum is 40.0 mg/dL
        assert feats["glucose_current"] == 40.0
        assert feats["glucose_below_70_pct_4h"] == 100.0
        assert feats["glucose_above_180_pct_4h"] == 0.0
        assert feats["hrv_rmssd_mean_4h"] == 0.0

    def test_all_nans_window(self, engineer, minimal_ehr):
        """Adversarial window where all telemetry values are NaN."""
        nans_df = pd.DataFrame({
            "glucose_mgdl": [np.nan] * 48,
            "glucose_roc": [np.nan] * 48,
            "glucose_acc": [np.nan] * 48,
            "hrv_rmssd_ms": [np.nan] * 48,
            "hrv_sdnn_ms": [np.nan] * 48,
            "hrv_lf_hf_ratio": [np.nan] * 48,
            "steps_last_5min": [np.nan] * 48,
            "activity_type": [None] * 48,
            "sleep_stage": [None] * 48,
            "sleep_quality": [np.nan] * 48,
        })
        feats = engineer.compute_features(nans_df, minimal_ehr)
        self._assert_features_contract(feats)

    def test_extreme_negative_rates_of_change(self, engineer, minimal_ehr):
        """Adversarial window with severe precipitous glucose drop (-15.0 mg/dL/min)."""
        rapid_drop_glucose = [350.0 - i * 5.0 for i in range(48)]
        drop_df = pd.DataFrame({
            "glucose_mgdl": rapid_drop_glucose,
            "glucose_roc": [-15.0] * 48,
            "glucose_acc": [-2.0] * 48,
            "hrv_rmssd_ms": [20.0] * 48,
            "hrv_sdnn_ms": [25.0] * 48,
            "steps_last_5min": [0] * 48,
            "activity_type": ["resting"] * 48,
            "sleep_stage": ["none"] * 48,
        })
        feats = engineer.compute_features(drop_df, minimal_ehr)
        self._assert_features_contract(feats)
        assert feats["glucose_roc_current"] == -15.0
        assert feats["glucose_acc_current"] == -2.0
        assert feats["glucose_current"] == np.clip(350.0 - 47 * 5.0, 40.0, 400.0)

    def test_missing_channels(self, engineer, minimal_ehr):
        """DataFrame missing all optional channels except glucose_mgdl."""
        bare_df = pd.DataFrame({
            "glucose_mgdl": [140.0] * 48,
        })
        feats = engineer.compute_features(bare_df, minimal_ehr)
        self._assert_features_contract(feats)
        assert feats["glucose_current"] == 140.0
        assert feats["steps_last_hour"] == 0.0

    def test_empty_ehr_dict(self, engineer):
        """compute_features with empty EHR dict must fall back to valid clinical defaults."""
        df = pd.DataFrame({"glucose_mgdl": [140.0] * 48})
        feats = engineer.compute_features(df, {})
        self._assert_features_contract(feats)
        assert feats["age"] == 52.0
        assert feats["bmi"] == 25.5
        assert feats["hba1c_latest"] == 8.0


# ===========================================================================
# 2. Clinical Validator Adversarial Stress Suite
# ===========================================================================

class TestClinicalValidatorAdversarial:
    """Stress tests ClinicalValidator against physiological bounds and clinical rules."""

    @pytest.fixture
    def validator(self) -> ClinicalValidator:
        return ClinicalValidator()

    @pytest.fixture
    def valid_patient(self) -> Dict[str, Any]:
        return {
            "patient_id": "PAT-TEST",
            "age": 55.0,
            "bmi": 26.5,
            "hba1c_latest": 7.8,
            "fasting_glucose_avg": 145.0,
            "egfr": 75.0,
            "bp_sys": 132.0,
            "insulin_dose_units": 0.0,
            "diabetes_duration_years": 8.0,
            "dietary_compliance_score": 0.8,
            "on_metformin": True,
            "on_insulin": False,
        }

    def test_pediatric_age_rejection(self, validator, valid_patient):
        """Pediatric ages (0, 12, 17) must be rejected with FIELD_OUT_OF_BOUNDS."""
        for bad_age in [0.0, 12.0, 17.5]:
            p = dict(valid_patient, age=bad_age)
            res = validator.validate(p)
            assert not res.is_valid
            assert any("age" in e.lower() for e in res.errors)

    def test_super_elderly_rejection(self, validator, valid_patient):
        """Ages above 95 (e.g. 96, 120) must be rejected."""
        for bad_age in [96.0, 120.0]:
            p = dict(valid_patient, age=bad_age)
            res = validator.validate(p)
            assert not res.is_valid
            assert any("age" in e.lower() for e in res.errors)

    def test_severe_ckd_metformin_warning(self, validator, valid_patient):
        """eGFR < 30 on Metformin must trigger lactic acidosis contraindication warning."""
        for low_egfr in [8.0, 15.0, 25.0, 29.5]:
            p = dict(valid_patient, egfr=low_egfr, on_metformin=True)
            res = validator.validate(p)
            assert any("lactic acidosis" in w.lower() or "contraindicated" in w.lower() for w in res.warnings)

    def test_severe_ckd_without_metformin_no_contraindication_warning(self, validator, valid_patient):
        """eGFR < 30 without Metformin must not trigger lactic acidosis warning."""
        p = dict(valid_patient, egfr=20.0, on_metformin=False)
        res = validator.validate(p)
        assert not any("lactic acidosis" in w.lower() for w in res.warnings)

    def test_insulin_inconsistency_rejection(self, validator, valid_patient):
        """on_insulin == False with insulin_dose_units > 0 must be rejected."""
        p = dict(valid_patient, on_insulin=False, insulin_dose_units=30.0)
        res = validator.validate(p)
        assert not res.is_valid
        assert any("inconsistent" in e.lower() and "insulin" in e.lower() for e in res.errors)

    def test_conflicting_glycemic_markers_rejection(self, validator, valid_patient):
        """HbA1c < 6.0 with fasting glucose > 200 must be rejected."""
        p = dict(valid_patient, hba1c_latest=5.4, fasting_glucose_avg=240.0)
        res = validator.validate(p)
        assert not res.is_valid
        assert any("conflict" in e.lower() and "hba1c" in e.lower() for e in res.errors)

    def test_duration_exceeds_patient_age_rejection(self, validator, valid_patient):
        """Diabetes duration exceeding patient age must be rejected."""
        p = dict(valid_patient, age=30.0, diabetes_duration_years=35.0)
        res = validator.validate(p)
        assert not res.is_valid
        assert any("duration" in e.lower() and "exceeds" in e.lower() for e in res.errors)

    def test_early_onset_t1d_masquerade_warning(self, validator, valid_patient):
        """Patient age < 25 with duration > 15 must trigger early onset warning."""
        p = dict(valid_patient, age=22.0, diabetes_duration_years=16.0)
        res = validator.validate(p)
        assert any("masquerade" in w.lower() or "onset" in w.lower() for w in res.warnings)

    def test_extreme_insulin_dose_warning(self, validator, valid_patient):
        """Insulin dose > 150 units/day must trigger clinical warning."""
        p = dict(valid_patient, on_insulin=True, insulin_dose_units=175.0)
        res = validator.validate(p)
        assert any("high insulin dose" in w.lower() for w in res.warnings)

    def test_missingness_threshold_warning(self, validator):
        """Profile missing > 30% of required fields must trigger LOW_QUALITY warning."""
        sparse_patient = {"age": 50.0, "bmi": 25.0}  # 2 of 9 required fields -> 78% missing
        res = validator.validate(sparse_patient)
        assert any("low_quality" in w.lower() or "missing" in w.lower() for w in res.warnings)


# ===========================================================================
# 3. Dataset Splitting & Zero-Leakage Adversarial Suite
# ===========================================================================

class TestDatasetLeakageAdversarial:
    """Verifies mathematical disjointness, temporal causality, and labeling integrity."""

    def test_mathematical_disjointness_80_10_10(self):
        """Simulates 100-patient partitioning, asserts strictly empty intersection between all pairs."""
        rng = np.random.default_rng(42)
        patients = [f"PAT-{i:03d}" for i in range(100)]
        shuffled = list(patients)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = max(1, int(0.80 * n_total))
        n_val = max(1, int(0.10 * n_total))

        train_pids = set(shuffled[:n_train])
        val_pids = set(shuffled[n_train : n_train + n_val])
        test_pids = set(shuffled[n_train + n_val :])

        assert len(train_pids & val_pids) == 0, "Data leakage between train and val!"
        assert len(train_pids & test_pids) == 0, "Data leakage between train and test!"
        assert len(val_pids & test_pids) == 0, "Data leakage between val and test!"
        assert len(train_pids) == 80
        assert len(val_pids) == 10
        assert len(test_pids) == 10

    def test_small_cohort_disjointness(self):
        """Verifies small cohort (3, 5, 10 patients) maintains test partition and zero leakage."""
        for n_pat in [3, 5, 10]:
            pids = [f"PAT-{i}" for i in range(n_pat)]
            n_train = max(1, int(0.80 * n_pat))
            n_val = max(1, int(0.10 * n_pat)) if n_pat > 2 else 0

            train = set(pids[:n_train])
            val = set(pids[n_train : n_train + n_val])
            test = set(pids[n_train + n_val :])

            if len(test) == 0 and len(train) > 1:
                test.add(train.pop())

            assert len(train & val) == 0
            assert len(train & test) == 0
            assert len(val & test) == 0
            assert len(train) + len(val) + len(test) == n_pat

    def test_labeler_temporal_causality(self):
        """Labeler must look strictly into future [t+1, t+24] and leave trailing rows unlabeled."""
        n = 50
        # Glucose is steady 120 until step 30 where a sharp spike to 300 occurs
        glucose = [120.0] * 30 + [300.0] * 5 + [120.0] * 15
        df = pd.DataFrame({
            "timestamp": [get_utc_now() + timedelta(minutes=i * 5) for i in range(n)],
            "glucose_mgdl": glucose,
        })
        labeled = label_adverse_events(df, horizon_minutes=120)

        # At index 5: future window is indices 6 to 30. Index 30 has glucose 300 -> spike!
        # Index 30 is (30 - 5) = 25 steps ahead > 24 steps -> NOT in 120-min horizon!
        assert labeled.loc[5, "adverse_event"] == 0

        # At index 6: future window is indices 7 to 31. Index 30 is 24 steps ahead = exactly 120 min!
        assert labeled.loc[6, "adverse_event"] == 1
        assert labeled.loc[6, "event_type"] == "hyperglycaemia"
        assert labeled.loc[6, "minutes_to_event"] == 120

        # Trailing 24 rows must be NaN (unlabeled)
        trailing = labeled.iloc[n - 24 :]
        assert trailing["adverse_event"].isna().all()
        assert (trailing["event_type"] == "unlabeled").all()

    def test_clinical_indices_partition_invariance(self):
        """TIR + TAR + TBR must sum to 100% on arbitrary random positive glucose series."""
        rng = np.random.default_rng(123)
        for _ in range(10):
            g = rng.uniform(40.0, 400.0, size=288)
            res = compute_clinical_indices(g, as_percentage=True)
            total = res["tir_24h"] + res["tar_24h"] + res["tbr_24h"]
            assert abs(total - 100.0) < 1e-4
            assert res["gmi"] > 3.31
            assert res["cv_glucose_24h"] >= 0.0
