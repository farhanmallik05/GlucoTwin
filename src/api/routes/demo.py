"""
src/api/routes/demo.py
Pre-configured Indian Patient Demographics Seeding Route for Hackathon Demonstrations.
"""

from fastapi import APIRouter, Depends
from src.api.middleware.auth import verify_api_key
from src.twin.engine import get_twin_engine

router = APIRouter(prefix="/api/v1/demo", tags=["Demo"])

DEMO_PATIENTS = {
    "DEMO-RAJESH": {
        "name": "Rajesh Kumar",
        "age": 52,
        "gender_encoded": 1,
        "bmi": 27.8,
        "hba1c_latest": 8.4,
        "hba1c_trend": 0.3,
        "fasting_glucose_avg": 165.0,
        "egfr": 78.0,
        "bp_sys": 138.0,
        "diabetes_duration_years": 7,
        "on_metformin": True,
        "on_insulin": True,
        "insulin_dose_units": 40.0,
        "on_sulphonylurea": False,
        "on_sglt2": True,
        "has_hypertension": True,
        "has_dyslipidaemia": True,
        "has_neuropathy": False,
        "has_retinopathy": False,
        "bmi_category_encoded": 2,
        "age_group_encoded": 2,
        "ethnicity_encoded": 1,
        "physical_activity_encoded": 1,
        "dietary_compliance_score": 0.65,
    },
    "DEMO-PRIYA": {
        "name": "Priya Sharma",
        "age": 46,
        "gender_encoded": 0,
        "bmi": 24.2,
        "hba1c_latest": 7.1,
        "hba1c_trend": -0.2,
        "fasting_glucose_avg": 118.0,
        "egfr": 92.0,
        "bp_sys": 122.0,
        "diabetes_duration_years": 4,
        "on_metformin": True,
        "on_insulin": False,
        "insulin_dose_units": 0.0,
        "on_sulphonylurea": True,
        "on_sglt2": False,
        "has_hypertension": False,
        "has_dyslipidaemia": False,
        "has_neuropathy": False,
        "has_retinopathy": False,
        "bmi_category_encoded": 1,
        "age_group_encoded": 1,
        "ethnicity_encoded": 1,
        "physical_activity_encoded": 2,
        "dietary_compliance_score": 0.85,
    },
    "DEMO-MOHAN": {
        "name": "Mohan Lal",
        "age": 61,
        "gender_encoded": 1,
        "bmi": 29.5,
        "hba1c_latest": 9.2,
        "hba1c_trend": 0.6,
        "fasting_glucose_avg": 185.0,
        "egfr": 58.0,
        "bp_sys": 145.0,
        "diabetes_duration_years": 12,
        "on_metformin": True,
        "on_insulin": True,
        "insulin_dose_units": 55.0,
        "on_sulphonylurea": False,
        "on_sglt2": False,
        "has_hypertension": True,
        "has_dyslipidaemia": True,
        "has_neuropathy": True,
        "has_retinopathy": True,
        "bmi_category_encoded": 2,
        "age_group_encoded": 3,
        "ethnicity_encoded": 1,
        "physical_activity_encoded": 0,
        "dietary_compliance_score": 0.50,
    },
    "DEMO-SUNITA": {
        "name": "Sunita Devi",
        "age": 39,
        "gender_encoded": 0,
        "bmi": 23.0,
        "hba1c_latest": 6.4,
        "hba1c_trend": -0.4,
        "fasting_glucose_avg": 105.0,
        "egfr": 105.0,
        "bp_sys": 115.0,
        "diabetes_duration_years": 2,
        "on_metformin": True,
        "on_insulin": False,
        "insulin_dose_units": 0.0,
        "on_sulphonylurea": False,
        "on_sglt2": False,
        "has_hypertension": False,
        "has_dyslipidaemia": False,
        "has_neuropathy": False,
        "has_retinopathy": False,
        "bmi_category_encoded": 1,
        "age_group_encoded": 1,
        "ethnicity_encoded": 1,
        "physical_activity_encoded": 2,
        "dietary_compliance_score": 0.90,
    },
}


@router.post("/seed", dependencies=[Depends(verify_api_key)])
async def seed_demo_patients():
    """Seed the 4 canonical Indian Type 2 Diabetes clinical cohorts into active memory."""
    engine = get_twin_engine()
    count = 0
    for pid, ehr in DEMO_PATIENTS.items():
        if engine.get_twin(pid) is None:
            engine.create_twin(pid, ehr)
        count += 1
    return {
        "status": "seeded",
        "count": count,
        "patients": list(DEMO_PATIENTS.keys())
    }
