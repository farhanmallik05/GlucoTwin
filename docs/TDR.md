# Technical Design Review (TDR)
## GlucoTwin — Type 2 Diabetes Digital Twin

| Document | Technical Design Review |
|---|---|
| **Version** | 1.0 |
| **Status** | Draft |
| **Author** | [Your Name] — Team Leader, Algo Alchemist |
| **Date** | October 2026 |
| **Project** | Digital Twin Challenge 2026 — Happiest Health |
| **Classification** | Public (Hackathon Submission) |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Data Architecture](#3-data-architecture)
4. [Digital Twin Engine Design](#4-digital-twin-engine)
5. [ML Model Architecture](#5-ml-model-architecture)
6. [API Design](#6-api-design)
7. [Frontend Architecture](#7-frontend-architecture)
8. [Security & Compliance](#8-security-compliance)
9. [Performance Specifications](#9-performance)
10. [Testing Strategy](#10-testing)
11. [Risks & Mitigations](#11-risks)
12. [Open Items](#12-open-items)

---

## 1. Executive Summary

GlucoTwin is a proof-of-concept Digital Twin system that creates a persistent, updateable virtual model of a Type 2 Diabetes patient. The system fuses static Electronic Health Record (EHR) data with dynamic wearable sensor streams to predict adverse glycaemic events — specifically glucose excursions (hyperglycaemia > 250 mg/dL or hypoglycaemia < 70 mg/dL) — **120 minutes before occurrence**.

### Design Goals

| Goal | Specification |
|---|---|
| Prediction Horizon | 120 minutes |
| Target AUC-ROC | ≥ 0.88 |
| Target Precision | ≥ 85% |
| API Latency | < 200ms (p95) |
| Data Privacy | DPDP Act compliant, no real patient data |
| Deployability | Single `docker-compose up` command |

---

## 2. System Overview

### 2.1 Component Topology

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GLUCOTWIN SYSTEM                              │
│                                                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │  EHR Layer  │    │ Wearable    │    │  Digital Twin Engine    │  │
│  │             │    │ Stream Layer│    │                          │  │
│  │ • Synthea   │───▶│             │───▶│ • PatientTwin class      │  │
│  │ • MIMIC-IV  │    │ • CGM Sim   │    │ • State management       │  │
│  │ • FHIR R4   │    │ • HRV Gen   │    │ • Physiological model    │  │
│  └─────────────┘    │ • Sleep API │    └────────────┬────────────┘  │
│                     └─────────────┘                 │               │
│                                                      ▼               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  Feature Engineering Pipeline                │    │
│  │  Rolling windows | Lag features | Clinical indices (TIR)    │    │
│  └──────────────────────────────┬──────────────────────────────┘    │
│                                 │                                    │
│                    ┌────────────┴────────────┐                      │
│                    ▼                         ▼                      │
│          ┌──────────────────┐    ┌──────────────────────┐           │
│          │  XGBoost Model   │    │   LSTM Model         │           │
│          │  (Tabular EHR)   │    │   (Time Series CGM)  │           │
│          └────────┬─────────┘    └──────────┬───────────┘           │
│                   └──────────┬───────────────┘                      │
│                              ▼                                       │
│                   ┌──────────────────────┐                          │
│                   │  Ensemble Predictor  │                          │
│                   │  + SHAP Explainer    │                          │
│                   └──────────┬───────────┘                          │
│                              │                                       │
│                              ▼                                       │
│                   ┌──────────────────────┐                          │
│                   │   FastAPI Backend    │                          │
│                   │   REST + WebSocket   │                          │
│                   └──────────┬───────────┘                          │
│                              │                                       │
│                              ▼                                       │
│                   ┌──────────────────────┐                          │
│                   │  React Dashboard     │                          │
│                   │  Doctor UI           │                          │
│                   └──────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack Justification

| Component | Choice | Rationale |
|---|---|---|
| EHR Generator | Synthea | Industry-standard, FHIR-native, T2D module built-in, free |
| Wearable Sim | Custom + Ohio T1DM | Ohio provides real CGM + physiological data; custom extends it |
| Tabular ML | XGBoost | Best-in-class for tabular clinical data; fast inference |
| Temporal ML | LSTM (PyTorch) | State-of-art for CGM time-series prediction |
| API | FastAPI | Async, type-safe, auto-generates OpenAPI docs |
| Frontend | React + Recharts | Mature ecosystem; Recharts excellent for clinical line charts |
| Container | Docker Compose | Reproducible deployment; judges can run locally |

---

## 3. Data Architecture

### 3.1 EHR Data Schema (Synthea Output → Processed)

```python
# Patient EHR Feature Vector
EHR_FEATURES = {
    # Demographics
    "age": int,                      # Years
    "gender": str,                   # M/F
    "bmi": float,                    # kg/m²
    "ethnicity": str,                # WHO categories

    # Clinical History
    "diabetes_duration_years": int,  # Years since T2D diagnosis
    "hba1c_latest": float,           # % (most recent value)
    "hba1c_trend": float,            # Change over 6 months
    "fasting_glucose_avg": float,    # mg/dL (30-day average)
    "egfr": float,                   # eGFR (kidney function)
    "blood_pressure_sys": float,     # mmHg

    # Medications
    "on_metformin": bool,
    "on_insulin": bool,
    "insulin_dose_units": float,     # 0 if not on insulin
    "on_sulphonylurea": bool,
    "on_sglt2": bool,

    # Comorbidities (binary flags)
    "has_hypertension": bool,
    "has_dyslipidaemia": bool,
    "has_neuropathy": bool,
    "has_retinopathy": bool,

    # Lifestyle
    "dietary_compliance_score": float,  # 0-1 (self-reported + EHR)
    "physical_activity_level": str,     # sedentary/light/moderate/active
}
```

### 3.2 Wearable Stream Schema (Real-time)

```python
# 5-minute interval wearable reading
WEARABLE_READING = {
    "patient_id": str,
    "timestamp": datetime,

    # CGM (Continuous Glucose Monitor)
    "glucose_mgdl": float,           # Current glucose (mg/dL)
    "glucose_rate_of_change": float, # mg/dL/min (CGM derivative)
    "glucose_acceleration": float,   # Second derivative

    # Heart Rate
    "heart_rate_bpm": float,
    "hrv_rmssd_ms": float,           # Root Mean Square of Successive Differences
    "hrv_sdnn_ms": float,

    # Activity
    "steps_last_5min": int,
    "steps_last_hour": int,
    "activity_type": str,            # resting/walking/exercise/sleep

    # Sleep (when applicable)
    "sleep_stage": str,              # awake/light/deep/rem
    "sleep_quality_score": float,    # 0-1

    # Derived
    "time_in_range_24h": float,      # % of last 24h glucose in 70-180 mg/dL
    "time_in_range_7d": float,
}
```

### 3.3 Feature Engineering — Prediction Features

The model uses **89 engineered features** derived from:

| Category | Features | Count |
|---|---|---|
| EHR Static | Raw EHR fields | 23 |
| CGM Statistics | Mean, std, min, max (1h, 4h, 12h, 24h) | 20 |
| CGM Derived | Rate of change, acceleration, excursion flags | 8 |
| HRV Features | RMSSD, SDNN, LF/HF ratio (rolling windows) | 12 |
| Activity Features | Steps, activity type encoding, calories | 8 |
| Sleep Features | Stage encoding, quality, duration | 6 |
| Temporal | Hour of day, day of week, meal proximity | 6 |
| Clinical Indices | TIR, TAR, TBR, GMI | 6 |
| **Total** | | **89** |

### 3.4 Data Sources & Licensing

| Dataset | Source | License | Usage |
|---|---|---|---|
| Synthetic EHR | Synthea (Mitre Corp) | Apache 2.0 | Training + Demo |
| CGM Time Series | Ohio T1DM Dataset | Research use | Training |
| Clinical Reference | MIMIC-IV (anonymised) | PhysioNet 1.5 | Validation |
| Synthetic Wearable | Custom generator | MIT | Demo + Training |

---

## 4. Digital Twin Engine

### 4.1 PatientTwin Class Design

```python
class PatientTwin:
    """
    Stateful virtual patient model — the core Digital Twin.
    Maintains physiological state and processes streaming sensor data.
    """

    # Identity
    patient_id: str
    twin_created_at: datetime
    twin_last_updated: datetime

    # Physiological State (updated each cycle)
    current_glucose: float           # Latest CGM reading
    glucose_trend: TrendDirection    # Rising/Stable/Falling
    estimated_insulin_on_board: float # Pharmacokinetic model
    circadian_phase: float           # 0-24 hour estimate

    # Personalised Baselines (learned from EHR + history)
    personal_glucose_mean: float     # Patient-specific normal
    personal_glucose_std: float
    insulin_sensitivity_factor: float
    carb_ratio: float

    # Risk State
    current_risk_score: float        # 0-1 probability of adverse event
    risk_tier: RiskTier              # GREEN/YELLOW/ORANGE/RED
    predicted_event_in: Optional[timedelta]
    last_alert: Optional[Alert]

    # History Buffer (sliding window)
    glucose_history: deque[float]    # Last 24 hours at 5-min intervals
    hrv_history: deque[float]
    activity_history: deque[str]

    def update(self, new_reading: WearableReading) -> PredictionResult:
        """Process new sensor reading and update twin state."""
        ...

    def predict(self) -> PredictionResult:
        """Run ensemble model on current state."""
        ...

    def get_shap_explanation(self) -> SHAPExplanation:
        """Explain current prediction with feature contributions."""
        ...
```

### 4.2 State Update Cycle

```
Every 5 minutes:
┌──────────────────────────────────────────────────────┐
│ 1. Receive new WearableReading                       │
│ 2. Update history buffers (glucose, HRV, activity)   │
│ 3. Recompute 89 engineered features                  │
│ 4. Run XGBoost + LSTM in parallel                    │
│ 5. Ensemble predictions → risk score                 │
│ 6. Update twin state (glucose, trend, risk)          │
│ 7. Check alert thresholds → fire if needed           │
│ 8. SHAP explanation if alert fired                   │
│ 9. Publish to WebSocket → dashboard updates          │
└──────────────────────────────────────────────────────┘
```

### 4.3 Alert System Design

| Tier | Condition | Action |
|---|---|---|
| 🟢 GREEN | Risk < 0.3 | Monitor normally |
| 🟡 YELLOW | 0.3 ≤ Risk < 0.6 | Dashboard indicator |
| 🟠 ORANGE | 0.6 ≤ Risk < 0.8 | Push notification to doctor |
| 🔴 RED | Risk ≥ 0.8 | Immediate alert + SHAP explanation |

---

## 5. ML Model Architecture

### 5.1 XGBoost Tabular Model

```
Input:  EHR static features + CGM aggregate statistics (89 features)
Output: P(adverse event in next 120 minutes)

Hyperparameters:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  scale_pos_weight: 4  (class imbalance correction)
  eval_metric: aucpr
```

### 5.2 LSTM Temporal Model

```
Input:  CGM sequence (last 4 hours × 5-min intervals = 48 timesteps)
        + HRV sequence (same window)
        + Activity sequence (same window)
        Shape: (batch, 48, 3)

Architecture:
  LSTM Layer 1: 128 units, return_sequences=True
  Dropout: 0.3
  LSTM Layer 2: 64 units, return_sequences=False
  Dropout: 0.2
  Dense: 32 units, ReLU
  Dense: 1 unit, Sigmoid

Output: P(adverse event in next 120 minutes)
```

### 5.3 Ensemble Strategy

```python
def ensemble_predict(xgb_prob: float, lstm_prob: float,
                     patient_risk_profile: str) -> float:
    """
    Weighted ensemble with patient-adaptive weights.
    High-insulin patients weight LSTM more (temporal patterns dominate).
    Non-insulin patients weight XGBoost more (EHR features dominate).
    """
    if patient_risk_profile == "insulin_dependent":
        w_xgb, w_lstm = 0.35, 0.65
    else:
        w_xgb, w_lstm = 0.60, 0.40

    return w_xgb * xgb_prob + w_lstm * lstm_prob
```

### 5.4 SHAP Explainability

Every RED alert generates a SHAP waterfall showing the **top 10 feature contributions**:

```
Example SHAP output for a RED alert:
  Base value: 0.23
  + glucose_rate_of_change_15min: +0.18
  + hrv_rmssd_drop_30min:         +0.14
  + time_since_last_meal:         +0.11
  + glucose_std_4h:               +0.09
  + insulin_on_board:             -0.07
  + steps_last_hour:              -0.04
  Final prediction: 0.84 (RED)
```

---

## 6. API Design

### 6.1 REST Endpoints

```
Base URL: http://localhost:8000/api/v1

POST   /patients              → Create new patient twin
GET    /patients/{id}         → Get twin state
POST   /patients/{id}/update  → Push new wearable reading
GET    /patients/{id}/predict → Get latest prediction
GET    /patients/{id}/history → Get glucose + alert history
GET    /patients/{id}/shap    → Get SHAP explanation for last alert
DELETE /patients/{id}         → Remove twin

GET    /health                → System health check
GET    /metrics               → Model performance metrics
```

### 6.2 WebSocket

```
WS: ws://localhost:8000/ws/patients/{id}

Emits every 5 minutes:
{
  "type": "twin_update",
  "patient_id": "...",
  "timestamp": "...",
  "glucose": 142.3,
  "trend": "rising",
  "risk_score": 0.72,
  "risk_tier": "ORANGE",
  "alert_fired": false,
  "top_features": [...]
}
```

### 6.3 Key Response Schemas

```python
class PredictionResponse(BaseModel):
    patient_id: str
    timestamp: datetime
    glucose_current: float
    glucose_trend: str
    risk_score: float           # 0.0 - 1.0
    risk_tier: str              # GREEN/YELLOW/ORANGE/RED
    predicted_event_in_minutes: Optional[int]
    confidence_interval: tuple[float, float]
    top_contributing_features: List[SHAPFeature]
    alert_message: Optional[str]
```

---

## 7. Frontend Architecture

### 7.1 Dashboard Components

```
App
├── PatientSelector        — Search + select active patient
├── TwinStatusBar         — Current glucose, trend arrow, risk badge
├── GlucoseTimeline       — Recharts LineChart (24h + prediction window)
│   ├── Historical line   — Blue
│   ├── Prediction band   — Orange shaded region
│   └── Alert markers     — Red/orange vertical lines
├── AlertPanel            — Current alerts with severity + SHAP
├── SHAPWaterfallChart    — Feature contribution bar chart
├── WearableMetrics       — HRV, steps, sleep stage cards
└── PatientEHRSummary     — Static demographics + medications
```

### 7.2 Real-time Update Flow

```
WebSocket message received
  → React state update (useWebSocket hook)
  → GlucoseTimeline re-renders with new point
  → Risk badge colour transitions (CSS animation)
  → Alert panel updates if alert fired
  → SHAP chart updates with new explanation
```

---

## 8. Security & Compliance

### 8.1 Data Privacy (DPDP Act Compliance)

| Requirement | Implementation |
|---|---|
| No real patient data | 100% synthetic (Synthea + custom generator) |
| Data minimisation | Only necessary features extracted |
| Patient anonymisation | UUID-based patient IDs, no PII |
| Audit trail | All predictions logged with timestamp |

### 8.2 API Security (PoC Level)

- API key authentication on all endpoints (header: `X-API-Key`)
- CORS configured for localhost only
- Input validation via Pydantic schemas
- Rate limiting: 100 requests/minute per key

> **Note**: Production deployment would require OAuth 2.0 + ABDM token integration.

---

## 9. Performance Specifications

| Metric | Target | Measurement Method |
|---|---|---|
| Prediction Latency | < 200ms (p95) | Locust load test |
| Throughput | 50 concurrent twins | Docker load test |
| Model AUC-ROC | ≥ 0.88 | Held-out test set |
| Model Precision | ≥ 85% | Held-out test set |
| Model Recall | ≥ 80% | Held-out test set |
| Dashboard FPS | ≥ 30 | Chrome DevTools |
| Memory per Twin | < 50MB | Docker stats |

---

## 10. Testing Strategy

### 10.1 Test Pyramid

```
Unit Tests (pytest)
├── test_patient_twin.py     — PatientTwin state transitions
├── test_feature_engineering.py — Feature computation
├── test_models.py           — Model load + inference shape
└── test_shap.py             — SHAP output validation

Integration Tests (pytest + httpx)
├── test_api_endpoints.py    — All REST endpoints
└── test_websocket.py        — WebSocket message flow

End-to-End Test (manual)
└── demo_patient_scenario.py — Full 24h simulation playback
```

### 10.2 Demo Test Scenario

The demo patient "Patient DEMO-001" is pre-seeded with a scenario:
- Hours 0-18: Normal glucose range (80-140 mg/dL)
- Hour 19: Post-meal spike building (trend: rapidly rising)
- Hour 20: Model fires ORANGE alert (risk 0.65)
- Hour 20.5: Model fires RED alert (risk 0.82)
- Hour 21: Glucose reaches 268 mg/dL (confirmed excursion)
- Hour 21+: Shows how pre-emptive action would have avoided the excursion

---

## 11. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Ohio T1DM dataset access issues | Medium | High | Backup: pure synthetic CGM via GluPy |
| LSTM training overfitting | Medium | High | Dropout + early stopping + validation curve |
| WebSocket disconnects in demo | Low | High | Auto-reconnect logic + fallback polling |
| Synthea Java installation failure | Low | Medium | Pre-generated dataset included in repo |
| Docker memory limits | Low | Medium | Resource limits in compose file |
| Model bias (gender/age) | Medium | Medium | Demographic parity analysis in notebook |

---

## 12. Open Items

| Item | Owner | Priority | Target Date |
|---|---|---|---|
| ABDM FHIR R4 connector design | [Your Name] | P2 | Post-hackathon |
| Real CGM device SDK integration | [Your Name] | P2 | Post-hackathon |
| Federated learning for model updates | [Your Name] | P3 | Future |
| NABH compliance checklist | [Your Name] | P1 | Before summit |
| Model bias audit (gender/age stratification) | [Your Name] | P1 | Day 7 |

---

*TDR Version 1.0 | GlucoTwin | Digital Twin Challenge 2026*
*Prepared by: [Your Name] | Last Updated: October 2026*
