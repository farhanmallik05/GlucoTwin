"""
src/api/routes/fhir.py
HL7 FHIR R4/R5 Interoperability Endpoint for Hospital Information Systems (HIS / EMR).
Enables seamless twin ingestion from standard hospital EHR data without manual dual-entry.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger("glucotwin.fhir")

from src.api.middleware.auth import verify_api_key
from src.twin.engine import get_twin_engine

router = APIRouter(prefix="/api/v1/fhir", tags=["FHIR Interoperability"])

# Standard LOINC Codes for Type 2 Diabetes Monitoring
LOINC_MAP = {
    "4548-4": "hba1c_latest",           # Hemoglobin A1c
    "17856-6": "hba1c_latest",          # Hemoglobin A1c in blood
    "1558-6": "fasting_glucose_avg",    # Fasting glucose [Mass/volume] in Serum or Plasma
    "2339-0": "current_glucose",        # Glucose [Mass/volume] in Blood (CGM or random)
    "8480-6": "bp_sys",                 # Systolic blood pressure
    "39156-5": "bmi",                   # Body Mass Index
    "33914-3": "egfr",                  # Glomerular filtration rate
}


def _extract_patient_features_from_bundle(bundle_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse HL7 FHIR Bundle and extract structured demographics & clinical biomarkers.
    Handles FHIR ResourceType: 'Bundle' containing 'Patient' and 'Observation' entries.
    """
    patient_id = "FHIR-PATIENT"
    name = "Anonymous Patient"
    gender = 1  # 1 = male, 0 = female
    age = 50

    # Default EHR features dictionary for twin initialization
    ehr: Dict[str, Any] = {
        "name": name,
        "age": age,
        "gender_encoded": gender,
        "bmi": 26.5,
        "hba1c_latest": 7.8,
        "hba1c_trend": 0.1,
        "fasting_glucose_avg": 145.0,
        "egfr": 80.0,
        "bp_sys": 130.0,
        "diabetes_duration_years": 5,
        "on_metformin": True,
        "on_insulin": False,
        "insulin_dose_units": 0.0,
        "on_sulphonylurea": False,
        "on_sglt2": False,
        "has_hypertension": False,
        "has_dyslipidaemia": False,
        "has_neuropathy": False,
        "has_retinopathy": False,
    }

    entries = bundle_data.get("entry", [])
    if not isinstance(entries, list):
        return {"patient_id": patient_id, "ehr_features": ehr, "cgm_readings": []}

    cgm_readings: List[Dict[str, Any]] = []

    for entry in entries:
        resource = entry.get("resource", {})
        res_type = resource.get("resourceType")

        if res_type == "Patient":
            if "id" in resource:
                patient_id = f"FHIR-{resource['id']}"
            if "name" in resource and len(resource["name"]) > 0:
                name_entry = resource["name"][0]
                given = " ".join(name_entry.get("given", []))
                family = name_entry.get("family", "")
                name = f"{given} {family}".strip() or name
                ehr["name"] = name
            if "gender" in resource:
                gender_str = resource["gender"].lower()
                gender = 0 if gender_str in ("female", "f") else 1
                ehr["gender_encoded"] = gender
            if "birthDate" in resource:
                try:
                    birth_year = int(resource["birthDate"].split("-")[0])
                    current_year = datetime.now(timezone.utc).year
                    age = max(18, min(100, current_year - birth_year))
                    ehr["age"] = age
                except Exception:
                    pass

        elif res_type == "Observation":
            code_obj = resource.get("code", {})
            codings = code_obj.get("coding", [])
            val_quantity = resource.get("valueQuantity", {})
            val_num = val_quantity.get("value")

            if val_num is not None:
                try:
                    val_float = float(val_num)
                    for c in codings:
                        code_val = c.get("code")
                        if code_val in LOINC_MAP:
                            mapped_feat = LOINC_MAP[code_val]
                            if mapped_feat == "current_glucose":
                                cgm_readings.append({
                                    "timestamp": resource.get("effectiveDateTime", datetime.now(timezone.utc).isoformat()),
                                    "glucose_mgdl": val_float
                                })
                            else:
                                ehr[mapped_feat] = val_float
                except (ValueError, TypeError):
                    pass

    return {
        "patient_id": patient_id,
        "ehr_features": ehr,
        "cgm_readings": cgm_readings
    }


@router.post("/patient-bundle", status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_api_key)])
async def ingest_fhir_patient_bundle(payload: Dict[str, Any]):
    """
    Ingest HL7 FHIR Bundle (Patient + Observations).
    Creates or updates the virtual patient twin and returns twin initialization status.
    Compatible with Epic, Cerner, Practo, and KareXpert EMR exports.
    """
    res_type = payload.get("resourceType")
    if res_type != "Bundle":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Payload must be a valid HL7 FHIR Bundle (resourceType: 'Bundle')."
        )

    parsed = _extract_patient_features_from_bundle(payload)
    pid = parsed["patient_id"]
    ehr = parsed["ehr_features"]
    cgm_list = parsed["cgm_readings"]

    engine = get_twin_engine()
    twin = engine.get_or_create_twin(pid, ehr)

    # Ingest any inline CGM observations attached to the bundle
    for cgm in cgm_list:
        from src.data.ingestion.wearable_simulator import WearableReading
        ts_val = cgm.get("timestamp")
        if isinstance(ts_val, str):
            try:
                ts = datetime.fromisoformat(ts_val.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                ts = datetime.now(timezone.utc).replace(tzinfo=None)
        else:
            ts = datetime.now(timezone.utc).replace(tzinfo=None)

        reading = WearableReading(
            patient_id=pid,
            timestamp=ts,
            glucose_mgdl=cgm["glucose_mgdl"],
            glucose_roc=0.0,
            hrv_rmssd_ms=35.0
        )
        twin.update(reading)

    logger.info(f"FHIR Bundle successfully ingested for patient {pid} with {len(cgm_list)} observations.")

    return {
        "status": "success",
        "message": f"Patient twin '{pid}' instantiated from HL7 FHIR Bundle.",
        "patient_id": pid,
        "demographics": {
            "name": ehr.get("name"),
            "age": ehr.get("age"),
            "hba1c": ehr.get("hba1c_latest"),
            "estimated_baseline_eag": round(twin.glucose_baseline_mean, 1)
        },
        "observations_ingested": len(cgm_list),
        "twin_summary": twin.get_summary()
    }
