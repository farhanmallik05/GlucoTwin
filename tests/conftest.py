"""
tests/conftest.py
Global pytest fixtures and mock test doubles for GlucoTwin test suite.
Supports progressive testability across all milestones (Tier 1 to Tier 4).
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to import domain classes; fall back gracefully if src is being implemented
try:
    from src.data.ingestion.wearable_simulator import WearableReading
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class WearableReading:
        patient_id: str
        timestamp: datetime
        glucose_mgdl: float
        glucose_roc: float
        glucose_acc: float = 0.0
        heart_rate_bpm: float = 75.0
        hrv_rmssd_ms: float = 38.0
        hrv_sdnn_ms: float = 45.0
        hrv_lf_hf_ratio: float = 1.5
        steps_last_5min: int = 20
        steps_last_hour: int = 150
        activity_type: str = "resting"
        calories_5min: float = 1.2
        sleep_stage: str = "none"
        sleep_quality: float = 0.8
        data_quality: str = "good"
        source: str = "cgm_sim"


# ---------------------------------------------------------------------------
# Fixture: Static EHR Profiles (23 Features)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ehr_features() -> Dict[str, Any]:
    """
    Standard sample EHR features for DEMO-RAJESH:
    52-year-old male, South Asian, non-insulin, HbA1c 8.4%, high risk of hyperglycemia.
    """
    return {
        "age": 52,
        "gender_encoded": 1,               # 1: Male, 0: Female
        "bmi": 27.2,
        "bmi_category_encoded": 2,         # Overweight (Indian cutoff)
        "ethnicity_encoded": 1,            # South Asian
        "diabetes_duration_years": 6,
        "hba1c_latest": 8.4,
        "hba1c_trend": 0.3,
        "fasting_glucose_avg": 158.0,
        "egfr": 82.0,
        "bp_sys": 138.0,
        "on_metformin": True,
        "on_insulin": False,
        "insulin_dose_units": 0.0,
        "on_sulphonylurea": False,
        "on_sglt2": True,
        "has_hypertension": True,
        "has_dyslipidaemia": False,
        "has_neuropathy": False,
        "has_retinopathy": False,
        "age_group_encoded": 1,            # 45-55
        "physical_activity_encoded": 0,    # Sedentary
        "dietary_compliance_score": 0.65,
    }


@pytest.fixture
def sample_ehr_features_insulin() -> Dict[str, Any]:
    """
    Sample EHR features for DEMO-MOHAN:
    58-year-old male, insulin-dependent (40 units/day), HbA1c 7.8%.
    """
    return {
        "age": 58,
        "gender_encoded": 1,
        "bmi": 28.5,
        "bmi_category_encoded": 3,         # Obese
        "ethnicity_encoded": 1,
        "diabetes_duration_years": 12,
        "hba1c_latest": 7.8,
        "hba1c_trend": -0.1,
        "fasting_glucose_avg": 145.0,
        "egfr": 68.0,
        "bp_sys": 142.0,
        "on_metformin": True,
        "on_insulin": True,
        "insulin_dose_units": 40.0,
        "on_sulphonylurea": True,
        "on_sglt2": False,
        "has_hypertension": True,
        "has_dyslipidaemia": True,
        "has_neuropathy": True,
        "has_retinopathy": False,
        "age_group_encoded": 2,            # 55-65
        "physical_activity_encoded": 1,    # Light
        "dietary_compliance_score": 0.55,
    }


@pytest.fixture
def sample_ehr_features_controlled() -> Dict[str, Any]:
    """
    Sample EHR features for DEMO-PRIYA:
    45-year-old female, well-controlled, HbA1c 6.5%, low risk.
    """
    return {
        "age": 45,
        "gender_encoded": 0,
        "bmi": 22.8,
        "bmi_category_encoded": 1,         # Normal
        "ethnicity_encoded": 1,
        "diabetes_duration_years": 3,
        "hba1c_latest": 6.5,
        "hba1c_trend": -0.4,
        "fasting_glucose_avg": 112.0,
        "egfr": 98.0,
        "bp_sys": 118.0,
        "on_metformin": True,
        "on_insulin": False,
        "insulin_dose_units": 0.0,
        "on_sulphonylurea": False,
        "on_sglt2": False,
        "has_hypertension": False,
        "has_dyslipidaemia": False,
        "has_neuropathy": False,
        "has_retinopathy": False,
        "age_group_encoded": 0,            # 35-45
        "physical_activity_encoded": 2,    # Moderate
        "dietary_compliance_score": 0.88,
    }


@pytest.fixture
def sample_ehr_features_brittle() -> Dict[str, Any]:
    """
    Sample EHR features for DEMO-SUNITA:
    68-year-old female, brittle elderly patient, CKD risk, hypoglycemia prone.
    """
    return {
        "age": 68,
        "gender_encoded": 0,
        "bmi": 21.0,
        "bmi_category_encoded": 1,         # Normal
        "ethnicity_encoded": 1,
        "diabetes_duration_years": 18,
        "hba1c_latest": 7.1,
        "hba1c_trend": 0.2,
        "fasting_glucose_avg": 130.0,
        "egfr": 42.0,
        "bp_sys": 150.0,
        "on_metformin": False,
        "on_insulin": True,
        "insulin_dose_units": 18.0,
        "on_sulphonylurea": False,
        "on_sglt2": False,
        "has_hypertension": True,
        "has_dyslipidaemia": True,
        "has_neuropathy": True,
        "has_retinopathy": True,
        "age_group_encoded": 3,            # 65+
        "physical_activity_encoded": 0,    # Sedentary
        "dietary_compliance_score": 0.70,
    }


# ---------------------------------------------------------------------------
# Fixture: Wearable Reading & Window Data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_wearable_reading() -> WearableReading:
    """Standard 5-minute single wearable reading."""
    return WearableReading(
        patient_id="DEMO-RAJESH",
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        glucose_mgdl=155.0,
        glucose_roc=1.2,
        glucose_acc=0.05,
        heart_rate_bpm=76.0,
        hrv_rmssd_ms=42.0,
        hrv_sdnn_ms=52.0,
        hrv_lf_hf_ratio=1.4,
        steps_last_5min=30,
        steps_last_hour=240,
        activity_type="resting",
        calories_5min=1.5,
        sleep_stage="none",
        sleep_quality=0.85,
        data_quality="good",
        source="cgm_sim"
    )


@pytest.fixture
def sample_window_df() -> pd.DataFrame:
    """
    4-hour sliding window DataFrame (48 timesteps at 5-min intervals).
    Contains required channels for FeatureEngineer.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    timestamps = [now - timedelta(minutes=(47 - i) * 5) for i in range(48)]
    np.random.seed(42)
    glucose = 140.0 + np.cumsum(np.random.normal(0.5, 2.0, 48))
    roc = np.gradient(glucose, 5.0)
    acc = np.gradient(roc, 5.0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "glucose_mgdl": glucose,
        "glucose_roc": roc,
        "glucose_acc": acc,
        "hrv_rmssd_ms": np.random.uniform(25.0, 50.0, 48),
        "hrv_sdnn_ms": np.random.uniform(35.0, 70.0, 48),
        "hrv_lf_hf_ratio": np.random.uniform(0.8, 2.5, 48),
        "steps_last_5min": np.random.randint(0, 100, 48),
        "steps_last_hour": np.random.randint(100, 800, 48),
        "activity_type": ["resting"] * 40 + ["walking"] * 8,
        "calories_5min": np.random.uniform(0.5, 4.0, 48),
        "sleep_stage": ["none"] * 48,
        "sleep_quality": [0.8] * 48,
        "data_quality": ["good"] * 48
    })
    return df


@pytest.fixture
def sample_24h_cgm_series() -> pd.Series:
    """24-hour continuous glucose series (288 readings at 5-minute intervals)."""
    np.random.seed(123)
    base = 130.0 + np.sin(np.linspace(0, 2 * np.pi, 288)) * 30.0
    noise = np.random.normal(0, 8.0, 288)
    return pd.Series(np.clip(base + noise, 40.0, 400.0))


# ---------------------------------------------------------------------------
# Fixture: Exactly 89 Tabular Features
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_89_features(sample_ehr_features) -> Dict[str, float]:
    """Generates an exact 89-key feature vector conforming to project contract."""
    try:
        from src.data.preprocessing.feature_engineering import FEATURE_COLUMNS
        static_cols = FEATURE_COLUMNS[:23]
    except ImportError:
        static_cols = [
            "age", "gender_encoded", "bmi", "hba1c_latest", "hba1c_trend", "fasting_glucose_avg",
            "egfr", "bp_sys", "diabetes_duration_years", "on_metformin", "on_insulin",
            "insulin_dose_units", "on_sulphonylurea", "on_sglt2", "has_hypertension",
            "has_dyslipidaemia", "has_neuropathy", "has_retinopathy", "bmi_category_encoded",
            "age_group_encoded", "ethnicity_encoded", "physical_activity_encoded", "dietary_compliance_score"
        ]
    features = {}
    # 1. Static EHR (23)
    for col in static_cols:
        val = sample_ehr_features.get(col, 0.0)
        features[col] = float(val) if isinstance(val, (int, float, bool)) else 0.0

    # 2. CGM Statistics (20)
    features.update({
        "glucose_mean_1h": 160.0, "glucose_std_1h": 12.0, "glucose_min_1h": 140.0, "glucose_max_1h": 180.0,
        "glucose_mean_4h": 155.0, "glucose_std_4h": 18.0, "glucose_min_4h": 130.0, "glucose_max_4h": 195.0,
        "glucose_mean_12h": 148.0, "glucose_std_12h": 22.0, "glucose_min_12h": 115.0, "glucose_max_12h": 210.0,
        "glucose_mean_24h": 142.0, "glucose_std_24h": 25.0,
        "glucose_roc_mean_30min": 1.1, "glucose_roc_std_30min": 0.4,
        "glucose_acc_mean_15min": 0.02,
        "glucose_percentile_10_4h": 135.0, "glucose_percentile_90_4h": 185.0, "glucose_iqr_4h": 25.0,
    })

    # 3. CGM Derived (8)
    features.update({
        "glucose_current": 165.0, "glucose_roc_current": 1.4, "glucose_acc_current": 0.05,
        "glucose_excursion_count_4h": 1.0, "glucose_above_180_pct_4h": 12.5, "glucose_below_70_pct_4h": 0.0,
        "glucose_coefficient_of_variation_4h": 11.6, "glucose_personal_deviation": 15.0,
    })

    # 4. HRV Features (12)
    features.update({
        "hrv_rmssd_mean_30min": 36.0, "hrv_rmssd_std_30min": 4.5, "hrv_rmssd_min_30min": 28.0,
        "hrv_rmssd_mean_4h": 40.0, "hrv_rmssd_trend_4h": -0.8,
        "hrv_sdnn_mean_1h": 50.0, "hrv_sdnn_std_1h": 6.2,
        "hrv_lf_hf_mean_1h": 1.6,
        "hrv_suppression_flag": 0.0, "hrv_personal_deviation": -4.0,
        "hrv_rmssd_pct_change_30min": -15.0, "hrv_recovery_score": 0.72,
    })

    # 5. Activity Features (8)
    features.update({
        "steps_last_hour": 320.0, "steps_last_4h": 1400.0, "steps_last_24h": 6500.0,
        "activity_type_encoded": 0.0, "is_exercising": 0.0, "is_sleeping": 0.0,
        "time_since_exercise_min": 180.0, "calories_last_hour": 75.0,
    })

    # 6. Sleep Features (6)
    features.update({
        "sleep_stage_encoded": 0.0, "sleep_quality_current": 0.85,
        "sleep_total_hours_last_night": 7.2, "sleep_efficiency_last_night": 0.88,
        "sleep_deep_pct_last_night": 22.0, "is_night_hours": 0.0,
    })

    # 7. Temporal Features (6)
    features.update({
        "hour_of_day": 14.0, "day_of_week": 2.0, "is_weekend": 0.0,
        "time_since_last_meal_min": 90.0, "time_since_last_insulin_min": 0.0,
        "circadian_phase": 0.58,
    })

    # 8. Clinical Indices (6)
    features.update({
        "tir_24h": 82.5, "tar_24h": 15.0, "tbr_24h": 2.5,
        "gmi": 6.70, "cv_glucose_24h": 17.6, "io_board": 0.0,
    })

    assert len(features) == 89, f"Expected 89 features, got {len(features)}"
    return features


# ---------------------------------------------------------------------------
# Fixture: Mock ML Models & Predictors
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_xgboost_model():
    """Mock XGBoost predictor with predict_proba and feature_importances_."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.25, 0.75]])
    model.feature_importances_ = np.random.uniform(0.01, 0.15, 89)
    model.is_fitted = True
    return model


@pytest.fixture
def mock_lstm_model():
    """Mock PyTorch LSTM temporal model returning excursion probability."""
    model = MagicMock()
    import torch
    model.return_value = torch.tensor([[0.70]], dtype=torch.float32)
    model.predict_proba.return_value = np.array([[0.30, 0.70]])
    return model


@pytest.fixture
def mock_shap_explainer():
    """Mock SHAP TreeExplainer returning top 10 clinical feature contributions."""
    explainer = MagicMock()
    explainer.explain.return_value = [
        {"feature": "glucose_roc_current", "clinical_label": "Glucose rate of change (+2.4 mg/dL/min)", "shap_value": 0.22, "direction": "increases_risk", "feature_value": 2.4},
        {"feature": "glucose_current", "clinical_label": "Current glucose (210 mg/dL)", "shap_value": 0.18, "direction": "increases_risk", "feature_value": 210.0},
        {"feature": "hrv_suppression_flag", "clinical_label": "Autonomic HRV suppression active", "shap_value": 0.14, "direction": "increases_risk", "feature_value": 1.0},
        {"feature": "hba1c_latest", "clinical_label": "Baseline HbA1c (8.4%)", "shap_value": 0.11, "direction": "increases_risk", "feature_value": 8.4},
        {"feature": "time_since_last_meal_min", "clinical_label": "Post-prandial window (45 min)", "shap_value": 0.09, "direction": "increases_risk", "feature_value": 45.0},
        {"feature": "hrv_rmssd_pct_change_30min", "clinical_label": "RMSSD drop (-35%)", "shap_value": 0.07, "direction": "increases_risk", "feature_value": -35.0},
        {"feature": "fasting_glucose_avg", "clinical_label": "Fasting average (158 mg/dL)", "shap_value": 0.05, "direction": "increases_risk", "feature_value": 158.0},
        {"feature": "bmi", "clinical_label": "BMI (27.2 kg/m²)", "shap_value": 0.03, "direction": "increases_risk", "feature_value": 27.2},
        {"feature": "age", "clinical_label": "Age (52 years)", "shap_value": 0.02, "direction": "increases_risk", "feature_value": 52.0},
        {"feature": "dietary_compliance_score", "clinical_label": "Diet compliance (0.65)", "shap_value": -0.04, "direction": "decreases_risk", "feature_value": 0.65},
    ]
    return explainer


# ---------------------------------------------------------------------------
# Fixture: Digital Twin Pre-Populated State
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_patient_twin(sample_ehr_features, sample_wearable_reading):
    """
    Stateful PatientTwin instance pre-populated with 288 readings (24 hours).
    Verifies that the twin is calibrated.
    """
    try:
        from src.twin.patient_twin import PatientTwin
    except ImportError:
        pytest.skip("src.twin.patient_twin not yet implemented")

    twin = PatientTwin(patient_id="DEMO-RAJESH", ehr_features=sample_ehr_features)
    base_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24)

    # Seed 288 readings
    for i in range(288):
        t = base_time + timedelta(minutes=i * 5)
        reading = WearableReading(
            patient_id="DEMO-RAJESH",
            timestamp=t,
            glucose_mgdl=135.0 + np.sin(i / 10.0) * 15.0,
            glucose_roc=0.1,
            glucose_acc=0.0,
            heart_rate_bpm=72.0,
            hrv_rmssd_ms=40.0,
            hrv_sdnn_ms=50.0,
            hrv_lf_hf_ratio=1.3,
            steps_last_5min=15,
            steps_last_hour=180,
            activity_type="resting",
            calories_5min=1.0,
            sleep_stage="none",
            sleep_quality=0.8,
            data_quality="good",
            source="cgm_sim"
        )
        twin.update(reading)

    return twin


def pytest_sessionfinish(session, exitstatus):
    """Ensure clean process exit by collecting garbage and closing background thread pools."""
    import gc
    gc.collect()
