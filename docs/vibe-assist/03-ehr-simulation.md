# Vibe Assist 03 — EHR Simulation
## Synthetic Patient Generation & Clinical Data Quality

> **Vibe Coding Session 03/10** | Estimated time: 3-4 hours
> Goal: Rich, clinically valid synthetic EHR data with realistic India-specific T2D profiles.

---

## 🎯 Session Goal

By the end of this session you will have:
- ✅ India-specific T2D patient demographics configured in Synthea
- ✅ EHR quality audit notebook showing data completeness
- ✅ Patient augmentation for edge cases (elderly, insulin-dependent, comorbid)
- ✅ Clinical validation checks ensuring realistic lab values
- ✅ FHIR → flat feature pipeline fully tested

---

## Clinical Context: India T2D Patient Profiles

Unlike Western populations, Indian T2D patients have distinct characteristics:

| Feature | Western T2D | Indian T2D |
|---|---|---|
| Average age at diagnosis | 55 years | 42 years (earlier onset) |
| BMI at diagnosis | Often >30 | Often 23-27 (lean T2D) |
| Insulin dependency | ~30% | ~45% (higher) |
| Comorbid hypertension | 60% | 75% |
| Dietary pattern | Mixed | High refined carb (rice/roti heavy) |
| HbA1c control | <7%: 50% | <7%: only 20% in India (ICMR) |
| Healthcare access | Urban: high | Rural: very limited |

Our synthetic patients must reflect these Indian demographics.

---

## 🤖 Vibe Prompt 1 — India-Specific Synthea Configuration

```
I need to configure Synthea to generate synthetic Indian T2D patients.

Create synthea_config/india_t2d_demographics.json that overrides 
Synthea's default US demographics with:

{
  "age_distribution": {
    "35-45": 0.30,  // Earlier onset in India
    "45-55": 0.35,
    "55-65": 0.25,
    "65+":   0.10
  },
  "gender": { "M": 0.52, "F": 0.48 },
  "bmi_distribution": {
    "18-23": 0.20,  // Lean T2D common in India
    "23-27": 0.40,
    "27-30": 0.25,
    "30+":   0.15
  },
  "hba1c_distribution": {
    "<7%":   0.20,  // Only 20% well-controlled (ICMR data)
    "7-8%":  0.25,
    "8-9%":  0.30,
    "9-10%": 0.15,
    ">10%":  0.10
  }
}

Also create synthea_config/t2d_medication_overrides.json that ensures:
- 45% of patients are on insulin (higher than US default)
- 80% are on metformin
- 30% are on SGLT2 inhibitors (empagliflozin/dapagliflozin)
- 25% are on sulfonylureas (glipizide)

Write a Python script (scripts/configure_synthea.py) that:
1. Modifies the Synthea synthea.properties to use these configs
2. Adds Indian city/state data: Mumbai, Delhi, Bengaluru, Chennai, 
   Hyderabad, Kolkata, Pune, Ahmedabad (weighted by T2D prevalence)
3. Outputs a summary of what was configured
```

---

## 🤖 Vibe Prompt 2 — EHR Data Quality Audit

```
Create a Jupyter notebook: notebooks/01_eda_ehr_data.ipynb

The notebook should perform a thorough EDA of the Synthea-generated 
EHR data (loaded from data/processed/ehr_features.csv):

Section 1: Dataset Overview
- Total patients, feature count
- Data types of each column
- Missing value heatmap (seaborn)
- Distribution of completeness per patient

Section 2: Demographics Analysis
- Age distribution histogram
- Gender pie chart
- BMI distribution with clinical category annotations (Underweight/Normal/
  Overweight/Obese class I/II/III)
- City/State distribution bar chart

Section 3: Clinical Features
- HbA1c distribution with risk zones colored:
  <7% = green, 7-8% = yellow, 8-9% = orange, >9% = red
- Fasting glucose distribution (log scale)
- eGFR distribution with CKD stage annotations
- Blood pressure distribution

Section 4: Medication Analysis
- Stacked bar: medication combinations
- Insulin dose distribution for insulin users
- Polypharmacy analysis (patients on 3+ medications)

Section 5: Comorbidity Analysis
- Comorbidity co-occurrence heatmap (Jaccard similarity)
- Venn diagram: hypertension ∩ dyslipidaemia ∩ neuropathy
- Comorbidity burden vs HbA1c scatter

Section 6: India-Specific Validation
- Compare generated stats vs ICMR 2023 reference values
- Flag any unrealistic values (e.g., age > 90, HbA1c > 15%)
- Data quality score per patient (0-100)

Use matplotlib + seaborn. Add markdown cells explaining each finding.
Clinical interpretation should reference real medical literature.
```

---

## 🤖 Vibe Prompt 3 — Patient Profile Edge Cases

```
I need to augment my synthetic patient dataset with specific edge cases 
that are important for model robustness.

Create src/data/synthetic/patient_generator.py with 
EdgeCaseGenerator class that creates 200 additional synthetic patients 
covering:

Edge Case 1: Lean T2D (BMI 20-23)
  - HbA1c: 8-10% (paradoxically high despite lean body)
  - High insulin secretion failure
  - Often younger (35-45 years)
  - Commonly seen in South India
  - 50 patients

Edge Case 2: Elderly Brittle Diabetic (age 70+)
  - Long diabetes duration (>20 years)
  - On insulin, high hypoglycaemia risk
  - eGFR < 60 (CKD stage 3)
  - Multiple comorbidities
  - 50 patients

Edge Case 3: Insulin-Naive but High HbA1c
  - HbA1c > 9% but no insulin
  - On oral medications only
  - Non-compliant patient profile
  - Frequent hyperglycaemia events
  - 50 patients

Edge Case 4: Well-Controlled on Insulin
  - HbA1c 6.5-7.0%
  - High hypoglycaemia risk due to tight control
  - Multiple daily injections
  - Active lifestyle
  - 50 patients

Each patient profile is a Python dict following the same EHR schema.
Output: data/synthetic/edge_case_patients.csv (200 patients)
Merge with main dataset: data/processed/ehr_features_augmented.csv (1200 patients)
```

---

## 🤖 Vibe Prompt 4 — Clinical Validation Layer

```
Create src/data/preprocessing/clinical_validator.py that validates 
synthetic patient EHR data against clinical reference ranges.

ClinicalValidator class with:

1. validate_patient(patient: dict) → ValidationResult
   Check each field against realistic clinical ranges:
   
   | Field | Min | Max | Notes |
   | age | 18 | 95 | |
   | bmi | 15 | 55 | Extremes are rare but possible |
   | hba1c | 4.0 | 15.0 | % |
   | fasting_glucose | 40 | 600 | mg/dL |
   | egfr | 5 | 120 | mL/min/1.73m² |
   | bp_systolic | 80 | 220 | mmHg |
   | insulin_dose | 0 | 200 | units/day |
   
   Clinical logic rules:
   - If on_insulin=False and insulin_dose_units > 0 → INCONSISTENT
   - If hba1c < 6.0 and fasting_glucose > 200 → INCONSISTENT  
   - If egfr < 30 and on_metformin=True → CLINICAL WARNING 
     (metformin contraindicated in severe CKD)
   - If age < 25 and diabetes_duration > 15 → VERIFY (T1D masquerade?)

2. validate_dataset(df: pd.DataFrame) → ValidationReport
   Aggregate validation across all patients.
   Report: % valid, % with warnings, % with errors.
   Return cleaned DataFrame (remove errors, flag warnings).

3. generate_validation_report() → HTML report
   A clean summary table showing validation results per field.

This validator ensures our synthetic data is medically realistic
and won't produce nonsensical predictions.
```

---

## 🤖 Vibe Prompt 5 — FHIR R4 to Flat Schema Mapper

```
Create src/data/ingestion/fhir_mapper.py that provides utilities for
working with FHIR R4 formatted data.

FHIRMapper class:

1. extract_patient_demographics(patient_resource: dict) → dict
   From FHIR Patient resource: extract birthDate → age, gender,
   extension[us-core-race], address.city, address.state

2. extract_conditions(condition_resources: List[dict]) → dict
   From FHIR Condition resources: map SNOMED codes to our boolean flags:
   - 44054006 = "Diabetes mellitus type 2" → is_diabetic
   - 59621000 = "Hypertension" → has_hypertension
   - 55822004 = "Dyslipidaemia" → has_dyslipidaemia
   - 230572002 = "Diabetic neuropathy" → has_neuropathy
   - 4855003 = "Diabetic retinopathy" → has_retinopathy

3. extract_observations(observation_resources: List[dict]) → dict
   From FHIR Observation resources: map LOINC codes:
   - 4548-4 = "HbA1c" → hba1c_latest + hba1c_trend (slope of last 2 values)
   - 1558-6 = "Fasting glucose" → fasting_glucose_avg
   - 33914-3 = "eGFR" → egfr
   - 55284-4 = "Blood pressure" → bp_systolic, bp_diastolic
   - 39156-5 = "BMI" → bmi

4. extract_medications(medication_resources: List[dict]) → dict
   From FHIR MedicationRequest: map RxNorm codes:
   - 860974 = Metformin → on_metformin
   - 311040 = Insulin Glargine → on_insulin, insulin_dose_units
   - 310489 = Glipizide → on_sulphonylurea
   - 2200653 = Empagliflozin → on_sglt2

5. Include a lookup table of all relevant codes as a class constant.
6. Log warnings for unrecognized FHIR codes.
7. Full unit tests in tests/test_fhir_mapper.py
```

---

## ✅ Validation: Clinical Realism Check

Run this to verify your EHR data matches ICMR 2023 statistics:

```python
# Quick sanity check script
import pandas as pd

df = pd.read_csv('data/processed/ehr_features.csv')

print("=== India T2D Population Validation ===")
print(f"Mean age: {df['age'].mean():.1f}y (ICMR: 47y)")
print(f"Mean BMI: {df['bmi'].mean():.1f} (ICMR: 25.8)")
print(f"Mean HbA1c: {df['hba1c_latest'].mean():.1f}% (ICMR: 8.6%)")
print(f"% on insulin: {df['on_insulin'].mean()*100:.0f}% (Target: ~45%)")
print(f"% hypertension: {df['has_hypertension'].mean()*100:.0f}% (Target: ~75%)")
print(f"% well-controlled (<7%): {(df['hba1c_latest']<7).mean()*100:.0f}% (ICMR: ~20%)")

# Red flags
assert df['hba1c_latest'].between(4, 15).all(), "HbA1c out of clinical range!"
assert df['age'].between(18, 95).all(), "Age out of range!"
assert df['bmi'].between(15, 55).all(), "BMI out of range!"
print("\n✅ All clinical validation checks passed!")
```

---

## 📋 Session 03 Completion Checklist

- [ ] Synthea configured with Indian demographics
- [ ] EDA notebook (01_eda_ehr_data.ipynb) complete with all sections
- [ ] Edge case patients generated (200 patients)
- [ ] Clinical validator running with 0 critical errors
- [ ] FHIR mapper handling all required LOINC/SNOMED/RxNorm codes
- [ ] Augmented dataset: 1200 patients × 23 features saved
- [ ] Sanity check script passes all assertions

## 🔜 Next Session

→ **04-wearable-integration.md** — Real-time wearable stream integration

---

*GlucoTwin Vibe Assist 03 | EHR Simulation | Digital Twin Challenge 2026*
