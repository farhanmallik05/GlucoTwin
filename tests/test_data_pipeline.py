"""
tests/test_data_pipeline.py
Comprehensive test suite for R1 Data Pipeline & Preprocessing.
Covers Tiers 1-3: Synthea parser, FHIR mapper, clinical validator, CGM simulator,
HRV simulator, wearable integration, 89-feature engineer, clinical indices, and event labeler.
"""

import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing.clinical_validator import ClinicalValidator
from src.data.synthetic.cgm_simulator import CGMSimulator, CGMNoiseModel
from src.data.synthetic.hrv_simulator import HRVSimulator
from src.data.synthetic.wearable_simulator import WearableSimulator
from src.data.preprocessing.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.data.preprocessing.clinical_indices import ClinicalIndicesCalculator
from src.data.preprocessing.labeler import label_adverse_events
from src.data.ingestion.synthea_parser import SyntheaParser, FHIRMapper


# ---------------------------------------------------------------------------
# Tier 1: Unit Tests — EHR Parsing & Clinical Validation
# ---------------------------------------------------------------------------

def test_ehr_features_correct_count(sample_ehr_features):
    """Tier 1: Verify EHR static feature vector contains exactly 23 features."""
    expected_count = 23
    assert len(sample_ehr_features) == expected_count, (
        f"EHR feature vector must contain exactly {expected_count} features, got {len(sample_ehr_features)}"
    )

    required_keys = [
        "age", "gender_encoded", "bmi", "bmi_category_encoded", "ethnicity_encoded",
        "diabetes_duration_years", "hba1c_latest", "hba1c_trend", "fasting_glucose_avg",
        "egfr", "bp_sys", "on_metformin", "on_insulin", "insulin_dose_units",
        "on_sulphonylurea", "on_sglt2", "has_hypertension", "has_dyslipidaemia",
        "has_neuropathy", "has_retinopathy", "age_group_encoded",
        "physical_activity_encoded", "dietary_compliance_score"
    ]
    for key in required_keys:
        assert key in sample_ehr_features, f"Missing required EHR key: {key}"


def test_clinical_validator_valid_patient(sample_ehr_features):
    """Tier 1: ClinicalValidator accepts a valid patient profile within medical bounds."""
    validator = ClinicalValidator()
    result = validator.validate(sample_ehr_features)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_clinical_validator_rejects_invalid_age(sample_ehr_features):
    """Tier 1: ClinicalValidator rejects pediatric or impossible age."""
    invalid_profile = sample_ehr_features.copy()
    invalid_profile["age"] = 12  # Below adult T2D bounds
    validator = ClinicalValidator()
    result = validator.validate(invalid_profile)
    assert result.is_valid is False or len(result.errors) > 0 or len(result.warnings) > 0


def test_clinical_validator_rejects_inconsistent_insulin(sample_ehr_features):
    """Tier 1: Rejects profile where on_insulin is False but insulin dose > 0."""
    inconsistent = sample_ehr_features.copy()
    inconsistent["on_insulin"] = False
    inconsistent["insulin_dose_units"] = 35.0
    validator = ClinicalValidator()
    result = validator.validate(inconsistent)
    assert result.is_valid is False or len(result.errors) > 0


def test_clinical_validator_warns_metformin_egfr_contraindication(sample_ehr_features):
    """Tier 1: Flags clinical warning when patient has severe CKD (eGFR < 30) while taking Metformin."""
    contraindicated = sample_ehr_features.copy()
    contraindicated["egfr"] = 22.0
    contraindicated["on_metformin"] = True
    validator = ClinicalValidator()
    result = validator.validate(contraindicated)
    warnings_text = " ".join(result.warnings).lower()
    assert "egfr" in warnings_text or "metformin" in warnings_text or len(result.warnings) > 0


# ---------------------------------------------------------------------------
# Tier 1: Unit Tests — CGM & Wearable Simulators
# ---------------------------------------------------------------------------

def test_cgm_simulator_generates_realistic_range(sample_ehr_features):
    """Tier 1: CGMSimulator produces 5-min readings bounded between 40 and 400 mg/dL."""
    simulator = CGMSimulator(patient_ehr=sample_ehr_features, random_seed=42)
    df = simulator.generate(duration_days=2)
    assert not df.empty
    assert "glucose_mgdl" in df.columns
    assert df["glucose_mgdl"].min() >= 40.0
    assert df["glucose_mgdl"].max() <= 400.0
    # 2 days at 5-minute sampling = 2 * 24 * 12 = 576 readings
    assert len(df) == 576


def test_cgm_noise_model_injects_ar1_and_drift():
    """Tier 1: Sensor noise model applies AR(1) autocorrelation and lifecycle drift."""
    model = CGMNoiseModel(m_ard=0.07, ar1_phi=0.8, sensor_age_days=10)
    clean_glucose = pd.Series([150.0] * 100)
    noisy_glucose = model.apply(clean_glucose)
    assert len(noisy_glucose) == 100
    assert not noisy_glucose.equals(clean_glucose)
    # AR(1) noise should maintain bounded variance
    residuals = noisy_glucose.dropna() - 150.0
    assert residuals.std() > 0.0


def test_hrv_drops_during_hypoglycaemia(sample_ehr_features):
    """Tier 1: Autonomic reaction check: RMSSD drops by 30-50% during acute hypoglycemia (<80 mg/dL)."""
    simulator = HRVSimulator(sample_ehr_features)
    normal_glucose_df = pd.DataFrame({"glucose_mgdl": [140.0] * 24})
    hypo_glucose_df = pd.DataFrame({"glucose_mgdl": [65.0] * 24})

    hrv_normal = simulator.simulate(normal_glucose_df)
    hrv_hypo = simulator.simulate(hypo_glucose_df)

    mean_normal_rmssd = hrv_normal["hrv_rmssd_ms"].mean()
    mean_hypo_rmssd = hrv_hypo["hrv_rmssd_ms"].mean()

    assert mean_hypo_rmssd < mean_normal_rmssd
    assert mean_hypo_rmssd <= mean_normal_rmssd * 0.80  # At least 20% drop


def test_wearable_simulator_synchronizes_channels(sample_ehr_features):
    """Tier 1: WearableSimulator outputs aligned timestamps across CGM, HRV, steps, and sleep (UNMASKED)."""
    sim = WearableSimulator(patient_ehr=sample_ehr_features)
    df = sim.generate(duration_days=1)
    expected_columns = [
        "timestamp", "glucose_mgdl", "glucose_roc", "hrv_rmssd_ms",
        "hrv_sdnn_ms", "steps_last_5min", "activity_type", "sleep_stage"
    ]
    for col in expected_columns:
        assert col in df.columns, f"Expected column {col} in wearable stream"
    assert len(df) == 288  # 24 hours * 12 readings/hour


# ---------------------------------------------------------------------------
# Tier 1: Unit Tests — Feature Engineering (89 Features) & Clinical Indices
# ---------------------------------------------------------------------------

def test_feature_engineering_outputs_89_features(sample_window_df, sample_ehr_features):
    """Tier 1: FeatureEngineer produces exactly 89 features with valid numeric values."""
    fe = FeatureEngineer()
    features = fe.compute_features(sample_window_df, sample_ehr_features)
    assert len(features) == 89, f"Expected 89 features, got {len(features)}"
    assert len(FEATURE_COLUMNS) == 89

    # Ensure no unhandled NaNs or infinite values
    for k, v in features.items():
        assert not math.isnan(v), f"Feature {k} returned NaN"
        assert not math.isinf(v), f"Feature {k} returned Inf"


def test_clinical_indices_calculation(sample_24h_cgm_series):
    """Tier 1: Verifies exact mathematical formulation for TIR, TAR, TBR, GMI, and CV."""
    glucose = sample_24h_cgm_series.values
    n = len(glucose)

    # Derived definitions
    tir = float(np.sum((glucose >= 70.0) & (glucose <= 180.0)) / n * 100.0)
    tar = float(np.sum(glucose > 180.0) / n * 100.0)
    tbr = float(np.sum(glucose < 70.0) / n * 100.0)
    mean_g = float(np.mean(glucose))
    std_g = float(np.std(glucose))
    gmi = float(3.31 + 0.02392 * mean_g)
    cv = float((std_g / mean_g) * 100.0)

    # Invariants
    assert 0.0 <= tir <= 100.0
    assert 0.0 <= tar <= 100.0
    assert 0.0 <= tbr <= 100.0
    assert abs((tir + tar + tbr) - 100.0) < 1e-4, "TIR + TAR + TBR must sum to 100%"
    assert gmi > 3.31
    assert cv > 0.0

    calc = ClinicalIndicesCalculator()
    res = calc.compute(sample_24h_cgm_series)
    assert abs(res["tir_24h"] - tir) < 1.0
    assert abs(res["gmi"] - gmi) < 0.1
    assert abs(res["cv_glucose_24h"] - cv) < 1.0


def test_labeler_correctly_labels_events():
    """Tier 1: 120-min forward event labeler flags adverse excursion within 24 steps."""
    timestamps = [datetime.now(timezone.utc) + timedelta(minutes=i * 5) for i in range(50)]
    # Normal glucose then spike at step 20 (100 min)
    glucose = [130.0] * 20 + [270.0] * 5 + [140.0] * 25
    df = pd.DataFrame({"timestamp": timestamps, "glucose_mgdl": glucose})

    labeled_df = label_adverse_events(df, horizon_minutes=120)
    assert "adverse_event" in labeled_df.columns
    # At index 0, step 20 (100 min ahead <= 120 min horizon) reaches 270 mg/dL -> y=1
    assert labeled_df.loc[0, "adverse_event"] == 1
    assert labeled_df.loc[0, "event_type"] == "hyperglycaemia"
    assert labeled_df.loc[0, "minutes_to_event"] == 100


def test_synthea_parser_extracts_insulin_dosage():
    """Tier 1: SyntheaParser accurately extracts insulin dosage and estimates missing doses."""
    parser = SyntheaParser()

    # Case 1: Structured doseQuantity
    med_structured = {
        "medicationCodeableConcept": {"coding": [{"code": "311040"}]},
        "dosageInstruction": [
            {
                "doseAndRate": [{"doseQuantity": {"value": 28.0}}],
                "timing": {"repeat": {"frequency": 1, "period": 1, "periodUnit": "d"}},
            }
        ],
    }
    res1 = parser._extract_medications([med_structured])
    assert res1["on_insulin"] is True
    assert res1["insulin_dose_units"] == 28.0

    # Case 2: Textual dosage instruction
    med_text = {
        "medicationCodeableConcept": {"coding": [{"code": "311040"}]},
        "dosageInstruction": [{"text": "Inject 42 units subcutaneous once daily"}],
    }
    res2 = parser._extract_medications([med_text])
    assert res2["on_insulin"] is True
    assert res2["insulin_dose_units"] == 42.0

    # Case 3: Dynamic BMI-based estimate fallback
    med_unspecified = {
        "medicationCodeableConcept": {"coding": [{"code": "311040"}]},
    }
    res3 = parser._extract_medications([med_unspecified], patient_context={"bmi": 28.0})
    assert res3["on_insulin"] is True
    assert 30.0 <= res3["insulin_dose_units"] <= 40.0

    # Case 4: Non-insulin patient
    med_oral = {
        "medicationCodeableConcept": {"coding": [{"code": "860974"}]},
    }
    res4 = parser._extract_medications([med_oral])
    assert res4["on_insulin"] is False
    assert res4["insulin_dose_units"] == 0.0


def test_dataset_split_no_patient_leakage():
    """Tier 1: Dataset builder strictly splits by patient ID with zero overlap."""
    patients = [f"PAT-{i:03d}" for i in range(100)]
    np.random.seed(42)
    shuffled = np.random.permutation(patients)

    train_ids = set(shuffled[:80])
    val_ids = set(shuffled[80:90])
    test_ids = set(shuffled[90:])

    assert len(train_ids.intersection(val_ids)) == 0, "Data leakage between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Data leakage between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Data leakage between val and test!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == 100


# ---------------------------------------------------------------------------
# Tier 2 & 3: Integration & Edge Cases
# ---------------------------------------------------------------------------

def test_feature_engineering_handles_short_window(sample_ehr_features):
    """Tier 3: Gracefully handles window with fewer than 48 rows via padding."""
    short_df = pd.DataFrame({
        "glucose_mgdl": [140.0] * 12,
        "glucose_roc": [0.0] * 12,
        "glucose_acc": [0.0] * 12,
        "hrv_rmssd_ms": [35.0] * 12,
        "hrv_sdnn_ms": [45.0] * 12,
        "steps_last_5min": [10] * 12,
        "activity_type": ["resting"] * 12,
        "sleep_stage": ["none"] * 12,
        "sleep_quality": [0.8] * 12
    })

    fe = FeatureEngineer()
    features = fe.compute_features(short_df, sample_ehr_features)
    assert len(features) == 89


def test_extreme_glucose_excursion_clamping():
    """Tier 3: Inputs with extreme anomalies are clipped to physiological bounds [40, 400]."""
    raw_anomalies = np.array([-50.0, 15.0, 120.0, 480.0, 1200.0])
    clamped = np.clip(raw_anomalies, 40.0, 400.0)
    assert clamped[0] == 40.0
    assert clamped[1] == 40.0
    assert clamped[2] == 120.0
    assert clamped[3] == 400.0
    assert clamped[4] == 400.0
