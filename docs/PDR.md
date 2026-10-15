# Preliminary Design Review (PDR)
## GlucoTwin — Type 2 Diabetes Digital Twin

| Document | Preliminary Design Review |
|---|---|
| **Version** | 1.0 |
| **Status** | Approved for Development |
| **Author** | [Your Name] — Team Leader, Algo Alchemist |
| **Date** | October 2026 |
| **Project** | Digital Twin Challenge 2026 — Happiest Health |
| **Review Gate** | Pre-Development / Concept Approval |

---

## Table of Contents

1. [Purpose & Scope](#1-purpose)
2. [Problem Analysis](#2-problem-analysis)
3. [Clinical Requirements](#3-clinical-requirements)
4. [Proposed Solution Concept](#4-proposed-solution)
5. [Alternative Approaches Considered](#5-alternatives)
6. [Data Strategy](#6-data-strategy)
7. [ML Approach Selection](#7-ml-approach)
8. [System Boundaries](#8-system-boundaries)
9. [Feasibility Assessment](#9-feasibility)
10. [Resource Plan](#10-resource-plan)
11. [Success Criteria](#11-success-criteria)
12. [Decision Log](#12-decisions)

---

## 1. Purpose & Scope

### 1.1 Purpose

This Preliminary Design Review (PDR) documents the **conceptual design decisions** made before implementation of GlucoTwin begins. It captures:
- The clinical problem being addressed
- Alternative solution approaches evaluated
- The selected technical approach with justification
- Feasibility assessment for a 14-day solo development timeline

### 1.2 Scope

**In Scope:**
- Synthetic/open-source data generation and processing
- Binary classification: adverse glucose event in next 120 minutes (Yes/No)
- Patient-specific Digital Twin state model
- Doctor-facing dashboard
- Hackathon prototype (PoC quality)

**Out of Scope:**
- Real patient data integration
- Medical device certification (CE/FDA/CDSCO)
- Production cloud deployment
- Mobile application
- ABDM integration (future roadmap only)

---

## 2. Problem Analysis

### 2.1 Clinical Context

**Type 2 Diabetes in India (2026 Context):**

| Statistic | Value | Source |
|---|---|---|
| Diagnosed diabetics in India | 101 million | ICMR 2023 |
| Pre-diabetics | 136 million | IDF 2023 |
| Annual cost of diabetes care | ₹8,800 crore | IDF |
| Emergency admissions for hypoglycaemia | 1 in 6 T2D patients/year | AIIMS data |
| Average time from event onset to care | 4-6 hours (rural India) | NHFS-5 |

### 2.2 Root Cause of Adverse Events

```
Why do glycaemic excursions occur unexpectedly?

Insulin dose errors ─────────────────────┐
Unplanned physical activity ─────────────┤
Missed/late meals ───────────────────────┤─▶ Glucose dysregulation ──▶ Adverse event
Illness/stress (cortisol spike) ─────────┤    (hypo < 70 or hyper > 250 mg/dL)
Drug interactions ───────────────────────┤
Poor sleep / HRV suppression ────────────┘

Current Standard of Care:
Patient feels symptoms → Goes to clinic/ER → Reactive treatment

GlucoTwin Approach:
Continuous monitoring → 120-min prediction → Proactive intervention
```

### 2.3 Opportunity Statement

> The convergence of affordable CGMs, wearable HRV monitors, and AI/ML makes it technically feasible to predict glycaemic excursions before they happen. India's ABDM infrastructure is ready to host such solutions. The only missing piece is the intelligent fusion layer — the Digital Twin.

### 2.4 Key Insights from Literature Review

| Insight | Implication for Design |
|---|---|
| CGM trend (rate of change) is most predictive of near-term glucose trajectory | Must compute first and second derivatives of CGM |
| HRV suppression precedes hypoglycaemia by 60-90 minutes | Include HRV as primary temporal feature |
| Post-meal prediction requires meal timestamp data | Include time-since-last-meal as a feature |
| Insulin-on-board (IOB) significantly modulates risk | Pharmacokinetic IOB model needed |
| Individual glucose variability (GV) patterns differ greatly | Personalized baseline required |

---

## 3. Clinical Requirements

### 3.1 Functional Clinical Requirements

| ID | Requirement | Priority |
|---|---|---|
| CR-01 | System SHALL predict glucose excursion events 120 minutes before occurrence | Must Have |
| CR-02 | System SHALL fuse EHR data with ≥2 distinct wearable data streams | Must Have |
| CR-03 | System SHALL define "adverse event" as glucose < 70 mg/dL or > 250 mg/dL | Must Have |
| CR-04 | System SHALL provide a clinician-readable alert with explanation | Must Have |
| CR-05 | System SHALL display a doctor-facing dashboard | Must Have |
| CR-06 | System SHALL personalise predictions to individual patient baselines | Should Have |
| CR-07 | System SHALL tier alerts by severity (not binary) | Should Have |
| CR-08 | System SHALL show time-in-range (TIR) statistics | Should Have |
| CR-09 | System SHALL support multiple simultaneous patient twins | Could Have |
| CR-10 | System SHALL provide audit log of all predictions | Could Have |

### 3.2 Non-Functional Clinical Requirements

| ID | Requirement | Value |
|---|---|---|
| NFR-01 | Prediction latency (sensor → alert) | < 5 minutes |
| NFR-02 | False positive rate | < 20% |
| NFR-03 | False negative rate (missed events) | < 15% |
| NFR-04 | Data privacy | No real PII; DPDP Act compliant |
| NFR-05 | Explainability | Clinician-readable SHAP output for every RED alert |

---

## 4. Proposed Solution Concept

### 4.1 Concept: The Stateful Digital Twin

The fundamental innovation of GlucoTwin vs. a simple ML predictor is **statefulness**. A Digital Twin:

1. **Remembers** the patient's history (not just last reading)
2. **Learns** the patient's personal glucose patterns over time
3. **Simulates** how glucose will evolve under different conditions
4. **Updates** its internal model as new data arrives

This mirrors the medical concept of a "longitudinal patient record" but adds predictive power.

### 4.2 Conceptual Architecture

```
STATIC LAYER (EHR):
  Synthea T2D Patient ──▶ Feature extraction ──▶ Patient Profile Vector

DYNAMIC LAYER (Wearable):
  CGM Simulator ──▶ Real-time stream ──▶ Sliding window buffer
  HRV Generator ──┘

TWIN ENGINE:
  Patient Profile × Window Buffer ──▶ 89 features ──▶ Ensemble Model

OUTPUT:
  Risk Score (0-1) ──▶ Tier (GREEN/YELLOW/ORANGE/RED) ──▶ Alert + SHAP
```

### 4.3 User Journey — Doctor Interaction

```
8:00 AM   Doctor opens GlucoTwin dashboard
          Sees 4 patients: 3 GREEN, 1 YELLOW
          YELLOW patient (Patient ID: T2D-042): Risk 0.45, trend: rising

9:15 AM   Dashboard transitions YELLOW → ORANGE
          Alert: "High risk of hyperglycaemia event in ~90 minutes"
          SHAP shows: post-breakfast CGM spike + HRV suppression

9:20 AM   Doctor calls patient: "Did you eat more than usual today?"
          Patient confirms: "Yes, had a big breakfast, skipped morning walk"
          Doctor advises: correction insulin dose + light walk

10:30 AM  Risk score drops to GREEN: preventive action successful

Without GlucoTwin:
          Patient would have reached 280 mg/dL by 10:45 AM
          ER visit, IV insulin, 4-hour observation
```

---

## 5. Alternative Approaches Considered

### 5.1 Option A: Simple Binary Classifier (Rejected)

| Aspect | Detail |
|---|---|
| Description | XGBoost classifier on snapshots of EHR + latest CGM |
| Pros | Simple, fast to build |
| Cons | No temporal modelling; no statefulness; not a "Digital Twin" |
| **Decision** | **Rejected** — does not meet the Digital Twin requirement |

### 5.2 Option B: Pure LSTM Time Series (Rejected)

| Aspect | Detail |
|---|---|
| Description | LSTM on CGM stream only |
| Pros | Excellent temporal modelling |
| Cons | Ignores rich EHR information; misses comorbidities, medications |
| **Decision** | **Rejected** — discards valuable static features |

### 5.3 Option C: Physiological Simulation Model (Rejected for PoC)

| Aspect | Detail |
|---|---|
| Description | Mechanistic ODE-based glucose-insulin model (Bergman Minimal Model) |
| Pros | Clinically grounded, interpretable |
| Cons | Requires clinical parameter estimation per patient; 14-day timeline insufficient |
| **Decision** | **Rejected for PoC** — included as future roadmap item |

### 5.4 Option D: XGBoost + LSTM Ensemble (Selected ✅)

| Aspect | Detail |
|---|---|
| Description | XGBoost for tabular EHR features + LSTM for CGM time series; ensemble with adaptive weights |
| Pros | Best of both worlds; achieves clinical accuracy targets; buildable in 14 days |
| Cons | Slightly more complex to train and deploy than single model |
| **Decision** | **Selected** — optimal balance of performance, explainability, and buildability |

---

## 6. Data Strategy

### 6.1 Data Acquisition Plan

```
Step 1: EHR Data (Day 1)
├── Install Synthea (Java, open-source)
├── Configure T2D patient module
├── Generate 1,000 synthetic T2D patients
└── Export FHIR JSON → convert to flat CSV

Step 2: Wearable Data (Day 2)
├── Download Ohio T1DM Dataset (CGM + wristband sensor)
│   Source: ohio1dm.engineering.case.edu
│   License: Research use, free for academic hackathon
├── Build custom CGM simulator for augmentation
└── Synthesize HRV, steps, sleep from statistical distributions

Step 3: Data Fusion (Day 3)
├── Align EHR patient profiles with wearable streams
├── Create patient_id linkage table
└── Build unified training dataset
```

### 6.2 Dataset Specifications

| Dataset | Size | Patients | Timespan |
|---|---|---|---|
| Synthea EHR | ~200MB (raw) | 1,000 synthetic | 10 years history |
| Ohio CGM (subset) | ~50MB | 12 real (anonymised) | 8 weeks each |
| Synthetic Wearable | ~100MB (generated) | 1,000 linked | 12 weeks |
| **Training Set** | ~80% | 800 patients | Full |
| **Validation Set** | ~10% | 100 patients | Full |
| **Test Set** | ~10% | 100 patients | Full |

### 6.3 Adverse Event Definition (Ground Truth Label)

```python
def label_adverse_event(glucose_series: pd.Series,
                         horizon_minutes: int = 120) -> pd.Series:
    """
    Label each timestep: 1 if glucose goes outside [70, 250] mg/dL
    within the next `horizon_minutes` minutes, else 0.
    """
    labels = []
    n = len(glucose_series)
    steps = horizon_minutes // 5  # 5-minute intervals

    for i in range(n):
        future_window = glucose_series[i : i + steps]
        event = ((future_window < 70) | (future_window > 250)).any()
        labels.append(int(event))

    return pd.Series(labels, index=glucose_series.index)
```

### 6.4 Class Imbalance Strategy

Expected class distribution:
- Normal (no event in next 120 min): ~85%
- Adverse event: ~15%

Mitigation:
- XGBoost: `scale_pos_weight = 85/15 ≈ 5.7`
- LSTM: Class-weighted loss function
- Evaluation: Focus on AUC-PR (not AUC-ROC) for imbalanced data

---

## 7. ML Approach Selection

### 7.1 Model Selection Matrix

| Criterion | Weight | XGBoost | LSTM | Random Forest | LightGBM |
|---|---|---|---|---|---|
| Tabular EHR performance | 30% | 9/10 | 5/10 | 8/10 | 9/10 |
| Temporal sequence modelling | 30% | 3/10 | 10/10 | 2/10 | 3/10 |
| Training speed (14-day timeline) | 20% | 9/10 | 6/10 | 8/10 | 9/10 |
| Explainability (SHAP support) | 20% | 10/10 | 6/10 | 7/10 | 10/10 |
| **Weighted Score** | | **7.8** | **6.7** | **6.0** | **7.8** |

**Decision**: XGBoost + LSTM ensemble captures both 7.8 + 6.7 strengths. LightGBM is equivalent to XGBoost but XGBoost has better SHAP integration.

### 7.2 Training Infrastructure

```
Hardware:  Google Colab Free (T4 GPU) for LSTM training
           Local CPU for XGBoost training
Time est:  XGBoost: ~20 minutes (1000 patients, 89 features)
           LSTM: ~45 minutes (Colab T4)
           Total training: < 2 hours
```

---

## 8. System Boundaries

### 8.1 What GlucoTwin IS

- A **PoC prototype** demonstrating Digital Twin concepts
- A **prediction system** using synthetic/open-source data
- A **research tool** for exploring glucose event forecasting
- A **hackathon demonstration** of feasible technology

### 8.2 What GlucoTwin IS NOT

- A **medical device** (not CE/FDA/CDSCO certified)
- A **replacement for clinical judgment**
- A **system for real patient data**
- A **production-ready deployment**

### 8.3 Disclaimers (Required in Dashboard)

```
⚠️ RESEARCH PROTOTYPE — NOT FOR CLINICAL USE
This system uses synthetic data only and has not been validated
on real patient populations. All predictions are for demonstration
purposes only. Do not use for clinical decision making.
```

---

## 9. Feasibility Assessment

### 9.1 Technical Feasibility

| Component | Feasibility | Risk | Notes |
|---|---|---|---|
| Synthea EHR generation | ✅ HIGH | Low | Well-documented, Java runtime only |
| CGM wearable simulation | ✅ HIGH | Low | Python, straightforward statistics |
| XGBoost model | ✅ HIGH | Low | Mature library, fast training |
| LSTM model | ✅ MEDIUM-HIGH | Medium | Needs GPU (Colab free) |
| FastAPI backend | ✅ HIGH | Low | Familiar Python ecosystem |
| React dashboard | ✅ MEDIUM | Medium | Solo dev bandwidth concern |
| Docker integration | ✅ HIGH | Low | Standard practice |

### 9.2 Timeline Feasibility (Solo Developer)

```
Critical Path Analysis:
  Data generation (2 days) ──▶ Feature engineering (1 day)
     ──▶ XGBoost training (1 day) ──▶ LSTM training (1 day)
     ──▶ Ensemble (0.5 day) ──▶ API (1.5 days)
     ──▶ Dashboard (2 days) ──▶ Integration (1 day)
     ──▶ Video + docs (2 days)
     = 12 days on critical path

Total allocated: 14 days → 2 days buffer
```

**Assessment: FEASIBLE with disciplined scope control.**

---

## 10. Resource Plan

### 10.1 Development Resources

| Resource | Cost | Notes |
|---|---|---|
| Google Colab (GPU) | Free | T4 GPU for LSTM |
| GitHub | Free | Public repo |
| Synthea | Free | Apache 2.0 |
| Ohio T1DM Dataset | Free | Academic research |
| React / FastAPI libraries | Free | Open source |
| Docker Desktop | Free | Personal use |

**Total external cost: ₹0**

### 10.2 Time Budget (Solo Developer)

| Phase | Hours |
|---|---|
| Data & EHR pipeline | 16h |
| Wearable simulation | 10h |
| ML modelling | 20h |
| FastAPI backend | 12h |
| React dashboard | 20h |
| Docker + integration | 8h |
| Video production | 8h |
| Documentation | 10h |
| **Total** | **104 hours (~7.4h/day for 14 days)** |

---

## 11. Success Criteria

### 11.1 Minimum Viable Submission (Must Achieve)

- [ ] Model achieves AUC-ROC ≥ 0.85 on test set
- [ ] Dashboard shows real-time glucose chart with prediction
- [ ] At least one demo patient triggers a RED alert during demo
- [ ] GitHub repo is public and accessible
- [ ] 20-minute video uploaded to YouTube (unlisted)
- [ ] Dual data stream fusion demonstrated

### 11.2 Winning Criteria (Target)

- [ ] Model achieves AUC-ROC ≥ 0.90
- [ ] SHAP waterfall chart visible in dashboard for every RED alert
- [ ] WebSocket real-time updates (< 1 second latency)
- [ ] Clean architecture diagram in repo
- [ ] All 10 vibe-assist docs completed and committed
- [ ] Docker Compose one-command deployment works

### 11.3 Go/No-Go Decision Gate

**Before Day 5 (end of data/model phase):**
- If model AUC-ROC < 0.78 → Switch to simpler feature engineering approach
- If LSTM training fails → Fall back to XGBoost-only with larger feature set
- If Synthea install fails → Use pre-generated dataset from backup

---

## 12. Decision Log

| Date | Decision | Rationale | Alternatives Rejected |
|---|---|---|---|
| Day 0 | Select Type 2 Diabetes | Largest India disease burden; best open datasets | Hypertension (less dramatic dashboard), CVD (insufficient T1 wearable data) |
| Day 0 | XGBoost + LSTM ensemble | Optimal balance for solo 14-day timeline | Pure LSTM (misses EHR features), pure XGBoost (no temporal) |
| Day 0 | Synthea for EHR | Free, FHIR-native, excellent T2D module | MIMIC-IV (complex to process), Manual CSV (not realistic) |
| Day 0 | FastAPI over Flask | Async support for WebSocket; auto OpenAPI docs | Flask (no native async), Django (too heavy) |
| Day 0 | React over Vue | Wider component ecosystem; Recharts charting | Vue (smaller community), Svelte (less job-relevant) |
| Day 0 | Docker Compose | Reproducible demo; judges can run locally | Manual install (unreliable in demo environment) |
| Day 0 | SHAP over LIME | Better XGBoost integration; faster computation | LIME (slower, less consistent) |

---

*PDR Version 1.0 | GlucoTwin | Digital Twin Challenge 2026*
*Author: [Your Name] | Status: Approved for Development | October 2026*
