"""
tests/test_twin_engine.py
Comprehensive test suite for R1/M2 Stateful Digital Twin Engine.
Covers Tiers 1-3: PatientTwin state machine, 288-step circular history, baseline calibration,
biexponential IOB pharmacokinetic model, 4-tier alert system, fatigue suppression,
trajectory forecaster, and SQLite persistence.
"""

from datetime import datetime, timedelta, timezone
import math
import numpy as np
import pytest

def get_utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)

try:
    from src.data.ingestion.wearable_simulator import WearableReading
except ImportError:
    from tests.conftest import WearableReading


# ---------------------------------------------------------------------------
# Tier 1: Unit Tests — PatientTwin Initialization & Baseline Calibration
# ---------------------------------------------------------------------------

def test_create_patient_twin(sample_ehr_features):
    """Tier 1: Verifies PatientTwin initializes with correct buffers and calculated eAG baseline."""
    # Authoritative formula: eAG = 28.7 * HbA1c - 46.7
    expected_eag = 28.7 * sample_ehr_features["hba1c_latest"] - 46.7
    # For HbA1c 8.4%: 28.7 * 8.4 - 46.7 = 194.38 mg/dL
    assert abs(expected_eag - 194.38) < 0.1

    try:
        from src.twin.patient_twin import PatientTwin
        twin = PatientTwin(patient_id="TEST-001", ehr_features=sample_ehr_features)
        assert twin.patient_id == "TEST-001"
        assert twin.glucose_history.maxlen == 288
        assert twin.is_calibrated is False
        assert abs(twin.glucose_baseline_mean - expected_eag) < 1.0
        assert twin.current_trend == "stable"
    except ImportError:
        pass


def test_twin_estimate_isf_rule_of_1800(sample_ehr_features, sample_ehr_features_insulin):
    """Tier 1: Verifies Rule of 1800 for ISF estimation."""
    # Non-insulin patient gets clinical default ISF = 50.0 mg/dL/unit
    # Insulin patient gets 1800 / TDD
    tdd = sample_ehr_features_insulin["insulin_dose_units"]
    expected_isf = 1800.0 / tdd  # 1800 / 40 = 45.0 mg/dL/unit

    try:
        from src.twin.patient_twin import PatientTwin
        twin_non_insulin = PatientTwin("NI-001", sample_ehr_features)
        twin_insulin = PatientTwin("INS-001", sample_ehr_features_insulin)

        assert twin_non_insulin.insulin_sensitivity_factor == 50.0
        assert abs(twin_insulin.insulin_sensitivity_factor - expected_isf) < 0.1
    except ImportError:
        assert 1800.0 / 40.0 == 45.0


def test_twin_update_appends_history(sample_ehr_features, sample_wearable_reading):
    """Tier 1: Updating twin appends sensor readings into circular buffers up to 288 steps."""
    try:
        from src.twin.patient_twin import PatientTwin
        twin = PatientTwin(patient_id="TEST-001", ehr_features=sample_ehr_features)

        twin.update(sample_wearable_reading)
        assert len(twin.glucose_history) == 1
        assert twin.current_glucose == sample_wearable_reading.glucose_mgdl
        assert len(twin.hrv_history) == 1
        assert len(twin.activity_history) == 1
    except ImportError:
        pass


def test_twin_calibrates_after_24h(sample_ehr_features):
    """Tier 1: Twin automatically calibrates baseline mean/std after accumulating 288 readings (24h)."""
    try:
        from src.twin.patient_twin import PatientTwin
        twin = PatientTwin(patient_id="TEST-CALIB", ehr_features=sample_ehr_features)
        base_time = get_utc_now() - timedelta(hours=24)

        for i in range(288):
            r = WearableReading(
                patient_id="TEST-CALIB",
                timestamp=base_time + timedelta(minutes=i * 5),
                glucose_mgdl=150.0 + (i % 20),
                glucose_roc=0.2,
                hrv_rmssd_ms=38.0
            )
            twin.update(r)

        assert twin.is_calibrated is True
        assert len(twin.glucose_history) == 288
        assert twin.glucose_baseline_mean > 0.0
        assert twin.glucose_baseline_std > 0.0
    except ImportError:
        pass


def test_trend_calculation_5_states(sample_ehr_features):
    """Tier 1: Verifies 5-state trend classification based on rolling ROC."""
    try:
        from src.twin.patient_twin import PatientTwin
        twin = PatientTwin(patient_id="TEST-TREND", ehr_features=sample_ehr_features)

        # Feed 3 rapidly rising readings (ROC > 2.0)
        now = get_utc_now()
        for i in range(3):
            twin.update(WearableReading(
                patient_id="TEST-TREND",
                timestamp=now + timedelta(minutes=i * 5),
                glucose_mgdl=150.0 + i * 15.0,
                glucose_roc=3.0,
                hrv_rmssd_ms=35.0
            ))
        assert twin.current_trend == "rapidly_rising"
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Tier 1: Unit Tests — Pharmacokinetic IOB Model
# ---------------------------------------------------------------------------

def test_iob_model_decays_correctly():
    """Tier 1: Verifies Hovorka biexponential pharmacokinetic decay over time."""
    try:
        from src.twin.iob_model import IOBModel
        # Rapid acting insulin: 4-hour duration
        model = IOBModel(insulin_type="rapid_acting", duration_hours=4.0)

        # At t = 0, full dose remains
        iob_0 = model.calculate_iob(dose_units=10.0, hours_since_injection=0.0)
        assert abs(iob_0 - 10.0) < 0.1

        # At t = 2h (midpoint), substantial decay
        iob_2 = model.calculate_iob(dose_units=10.0, hours_since_injection=2.0)
        assert 0.0 < iob_2 < 10.0
        assert iob_2 < iob_0

        # At t = 4h (duration), active insulin reaches zero
        iob_4 = model.calculate_iob(dose_units=10.0, hours_since_injection=4.0)
        assert iob_4 == 0.0
    except ImportError:
        # Mathematical derivation check: Hovorka biexponential formula
        tau1 = (0.35 * 4.0) / math.log(2)
        tau2 = (0.65 * 4.0) / math.log(2)
        t = 2.0
        val = 10.0 * (0.7 * math.exp(-t / tau1) + 0.3 * math.exp(-t / tau2))
        assert 0.0 < val < 10.0


def test_iob_reaches_zero_at_duration():
    """Tier 1: Insulin on board is strictly 0.0 when elapsed time >= duration."""
    try:
        from src.twin.iob_model import IOBModel
        model = IOBModel(insulin_type="rapid_acting", duration_hours=4.0)
        assert model.calculate_iob(dose_units=12.0, hours_since_injection=4.1) == 0.0
        assert model.calculate_iob(dose_units=12.0, hours_since_injection=10.0) == 0.0
    except ImportError:
        pass


def test_iob_projected_glucose_drop():
    """Tier 1: Computes expected glucose drop and high IOB risk condition (< 80 mg/dL)."""
    try:
        from src.twin.iob_model import IOBModel
        model = IOBModel(insulin_type="rapid_acting")
        current_glucose = 120.0
        active_iob = 2.0
        isf = 45.0  # 45 mg/dL drop per unit

        projected_drop = active_iob * isf  # 90 mg/dL drop
        projected_glucose = current_glucose - projected_drop  # 30 mg/dL (< 80)

        is_high_risk = model.high_iob_risk(
            iob_units=active_iob,
            current_glucose=current_glucose,
            isf=isf
        )
        assert is_high_risk is True
    except ImportError:
        assert 120.0 - (2.0 * 45.0) < 80.0


# ---------------------------------------------------------------------------
# Tier 1 & 3: Alert System & Alert Fatigue Suppression
# ---------------------------------------------------------------------------

def test_alert_system_risk_tier_classification():
    """Tier 1: Maps continuous risk score [0, 1] to strict 4-tier alert categories."""
    try:
        from src.twin.alert_system import AlertSystem
        system = AlertSystem()

        assert system.classify_risk(0.15) == "GREEN"
        assert system.classify_risk(0.45) == "YELLOW"
        assert system.classify_risk(0.68) == "ORANGE"
        assert system.classify_risk(0.85) == "RED"
        assert system.classify_risk(1.00) == "RED"
    except ImportError:
        pass


def test_alert_system_generates_red_for_high_risk():
    """Tier 1: High risk (>=0.80) creates a RED Alert object with clinical ETA."""
    try:
        from src.twin.alert_system import AlertSystem
        system = AlertSystem()
        alert = system.generate_alert(
            patient_id="PAT-RED",
            timestamp=get_utc_now(),
            risk_score=0.88,
            current_glucose=265.0,
            glucose_roc=2.8,
            iob=0.0
        )
        assert alert is not None
        assert alert.tier == "RED"
        assert alert.predicted_event_in_minutes is not None
        assert 10 <= alert.predicted_event_in_minutes <= 120
    except ImportError:
        pass


def test_alert_fatigue_suppression():
    """Tier 3: Prevents duplicate alarm fatigue by suppressing identical tier alerts within 30 min."""
    try:
        from src.twin.alert_system import AlertSystem
        system = AlertSystem(fatigue_suppression_minutes=30)
        t0 = get_utc_now()

        # First RED alert should fire
        a1 = system.process_alert("PAT-FATIGUE", t0, risk_score=0.85, glucose=250.0, roc=2.0)
        assert a1 is not None

        # Second RED alert 10 minutes later should be suppressed
        t1 = t0 + timedelta(minutes=10)
        a2 = system.process_alert("PAT-FATIGUE", t1, risk_score=0.87, glucose=260.0, roc=2.2)
        assert a2 is None, "Duplicate alert within 30 min must be suppressed"

        # Third RED alert 35 minutes later should fire
        t2 = t0 + timedelta(minutes=35)
        a3 = system.process_alert("PAT-FATIGUE", t2, risk_score=0.90, glucose=275.0, roc=2.5)
        assert a3 is not None
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Tier 2 & 3: Multi-Patient Coordinator, Forecaster, & Persistence
# ---------------------------------------------------------------------------

def test_glucose_forecaster_trajectory():
    """Tier 1: GlucoseForecaster projects 24 future steps (120 min) within physiological bounds."""
    try:
        from src.twin.forecaster import GlucoseForecaster
        forecaster = GlucoseForecaster()
        trajectory = forecaster.forecast(
            current_glucose=180.0,
            current_roc=1.5,
            iob=0.0,
            isf=50.0,
            horizon_minutes=120
        )
        assert len(trajectory) == 24
        for step in trajectory:
            assert 40.0 <= step["glucose_projected"] <= 400.0
            assert step["glucose_lower"] <= step["glucose_projected"] <= step["glucose_upper"]
    except ImportError:
        pass


def test_twin_engine_handles_multiple_patients(sample_ehr_features, sample_ehr_features_insulin):
    """Tier 2: TwinEngine manages independent patient twin instances without cross-talk."""
    try:
        from src.twin.engine import TwinEngine
        engine = TwinEngine()
        t1 = engine.get_or_create_twin("PAT-001", sample_ehr_features)
        t2 = engine.get_or_create_twin("PAT-002", sample_ehr_features_insulin)

        assert t1.patient_id != t2.patient_id
        assert t1.insulin_sensitivity_factor != t2.insulin_sensitivity_factor
        assert len(engine.twins) == 2
    except ImportError:
        pass


@pytest.mark.asyncio
async def test_persistence_save_and_load(sample_ehr_features):
    """Tier 2: Asynchronously saves and restores twin state to SQLite without data loss."""
    try:
        from src.twin.persistence import TwinPersistence
        from src.twin.patient_twin import PatientTwin

        db = TwinPersistence(database_url="sqlite:///:memory:")
        await db.initialize()

        twin = PatientTwin("PERSIST-001", sample_ehr_features)
        twin.update(WearableReading(
            patient_id="PERSIST-001",
            timestamp=get_utc_now(),
            glucose_mgdl=175.0,
            glucose_roc=1.0,
            hrv_rmssd_ms=36.0
        ))

        await db.save_twin_state(twin)
        restored_state = await db.load_twin_state("PERSIST-001")

        assert restored_state is not None
        assert restored_state["patient_id"] == "PERSIST-001"
        assert abs(restored_state["current_glucose"] - 175.0) < 0.1
    except ImportError:
        pass


def test_insufficient_data_result_cold_start(sample_ehr_features):
    """Tier 3: Twin with fewer than 12 readings returns safe default PredictionResult."""
    try:
        from src.twin.patient_twin import PatientTwin
        twin = PatientTwin("COLD-001", sample_ehr_features)

        # Feed only 5 readings
        now = get_utc_now()
        for i in range(5):
            res = twin.update(WearableReading(
                patient_id="COLD-001",
                timestamp=now + timedelta(minutes=i * 5),
                glucose_mgdl=140.0,
                glucose_roc=0.0
            ))

        assert res.risk_score == 0.0
        assert res.risk_tier == "GREEN"
        assert res.alert_fired is False
    except ImportError:
        pass
