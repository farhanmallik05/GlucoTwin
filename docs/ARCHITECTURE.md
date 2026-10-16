# System Architecture
## GlucoTwin — Type 2 Diabetes Digital Twin

| Document | System Architecture |
|---|---|
| **Version** | 1.0 |
| **Author** | Farhan Mallick (Algo Alchemist) |
| **Date** | October 2026 |

---

## 1. Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════╗
║                    GLUCOTWIN SYSTEM ARCHITECTURE                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  DATA SOURCES LAYER                                                    ║
║  ┌─────────────────────┐    ┌────────────────────────────────────┐    ║
║  │   EHR / STATIC      │    │   WEARABLE / DYNAMIC               │    ║
║  │                     │    │                                    │    ║
║  │  • Synthea T2D      │    │  • CGM Simulator (5-min)           │    ║
║  │    (1000 patients)  │    │  • HRV Generator                   │    ║
║  │  • MIMIC-IV subset  │    │  • Step Counter Sim                │    ║
║  │  • FHIR R4 format   │    │  • Sleep Stage Model               │    ║
║  └─────────┬───────────┘    └────────────────┬───────────────────┘    ║
║            │                                 │                         ║
║  ─────────────────────────────────────────────────────────────────    ║
║  INGESTION & PROCESSING LAYER                                          ║
║            │                                 │                         ║
║            ▼                                 ▼                         ║
║  ┌─────────────────────────────────────────────────────────────┐      ║
║  │              Feature Engineering Pipeline                   │      ║
║  │                                                             │      ║
║  │  EHR Parser → Demographic + Clinical Features (23)         │      ║
║  │  CGM Processor → Statistics + Derivatives (28)             │      ║
║  │  HRV Processor → Time/Frequency Domain Features (12)       │      ║
║  │  Activity Processor → Steps + Sleep Features (14)          │      ║
║  │  Temporal Encoder → Clock + Meal Features (6)              │      ║
║  │  Clinical Indices → TIR, TAR, TBR, GMI (6)                │      ║
║  │                                           Total: 89 features│      ║
║  └─────────────────────────┬───────────────────────────────────┘      ║
║                             │                                           ║
║  ─────────────────────────────────────────────────────────────────    ║
║  DIGITAL TWIN ENGINE LAYER                                             ║
║                             │                                           ║
║                             ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────┐      ║
║  │                   PatientTwin (per patient)                 │      ║
║  │                                                             │      ║
║  │  State: { glucose, hrv, activity, risk_score, alert_hist } │      ║
║  │  Baselines: { personal_mean, sensitivity_factor, circadian }│      ║
║  │  IOB Model: pharmacokinetic insulin-on-board estimate       │      ║
║  └──────────────┬──────────────────────────┬────────────────────┘     ║
║                 │                          │                            ║
║  ─────────────────────────────────────────────────────────────────    ║
║  ML INFERENCE LAYER                                                    ║
║                 ▼                          ▼                            ║
║  ┌──────────────────────┐    ┌─────────────────────────────────┐      ║
║  │   XGBoost Model      │    │        LSTM Model               │      ║
║  │                      │    │                                  │      ║
║  │  Input: 89 tabular   │    │  Input: (48, 3) time series      │      ║
║  │  Output: P(event)    │    │  Output: P(event)               │      ║
║  │  Latency: ~15ms      │    │  Latency: ~80ms                 │      ║
║  └──────────┬───────────┘    └──────────────┬──────────────────┘      ║
║             │                               │                           ║
║             └───────────────┬───────────────┘                          ║
║                             │                                           ║
║                             ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────┐      ║
║  │                Ensemble + SHAP Layer                        │      ║
║  │                                                             │      ║
║  │  Weighted ensemble → Final risk score (0-1)                │      ║
║  │  Risk tiering: GREEN / YELLOW / ORANGE / RED               │      ║
║  │  SHAP explainer → Top 10 feature contributions             │      ║
║  │  Alert generator → Message + severity + time-to-event      │      ║
║  └──────────────────────────┬──────────────────────────────────┘      ║
║                             │                                           ║
║  ─────────────────────────────────────────────────────────────────    ║
║  API LAYER                                                             ║
║                             │                                           ║
║                             ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────┐      ║
║  │                  FastAPI Backend                            │      ║
║  │                                                             │      ║
║  │  REST:      /api/v1/patients/* (CRUD + predict)            │      ║
║  │  WebSocket: ws://*/ws/patients/{id} (5-min push)           │      ║
║  │  Middleware: API key auth, CORS, rate limiting              │      ║
║  │  Storage:   SQLite (patient state + prediction history)     │      ║
║  └──────────────────────────┬──────────────────────────────────┘      ║
║                             │                                           ║
║  ─────────────────────────────────────────────────────────────────    ║
║  PRESENTATION LAYER                                                    ║
║                             │                                           ║
║                             ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────┐      ║
║  │               React Dashboard (Doctor UI)                   │      ║
║  │                                                             │      ║
║  │  PatientSelector → Patient list with risk badges           │      ║
║  │  TwinStatusBar → Live glucose + trend + risk tier          │      ║
║  │  GlucoseTimeline → 24h CGM chart + 2h prediction band      │      ║
║  │  AlertPanel → Active alerts with SHAP explanation          │      ║
║  │  WearableMetrics → HRV, steps, sleep cards                 │      ║
║  │  EHRSummary → Demographics + medications                   │      ║
║  └─────────────────────────────────────────────────────────────┘      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 2. Directory Structure

```
GlucoTwin/
│
├── 📄 README.md                    ← Submission README
├── 📄 LICENSE                      ← MIT License
├── 📄 .gitignore
├── 📄 docker-compose.yml           ← One-command deployment
├── 📄 requirements.txt             ← Python dependencies
│
├── 📁 docs/
│   ├── 📄 HACKATHON_GUIDE.md       ← Winning strategy
│   ├── 📄 TDR.md                   ← Technical Design Review
│   ├── 📄 PDR.md                   ← Preliminary Design Review
│   ├── 📄 ARCHITECTURE.md          ← This file
│   ├── 📄 architecture_diagram.pdf ← Visual architecture
│   ├── 📄 presentation.pdf         ← Summit slides
│   └── 📁 vibe-assist/
│       ├── 📄 01-project-setup.md
│       ├── 📄 02-data-pipeline.md
│       ├── 📄 03-ehr-simulation.md
│       ├── 📄 04-wearable-integration.md
│       ├── 📄 05-digital-twin-engine.md
│       ├── 📄 06-ml-model-training.md
│       ├── 📄 07-prediction-engine.md
│       ├── 📄 08-api-backend.md
│       ├── 📄 09-dashboard-ui.md
│       └── 📄 10-deployment-testing.md
│
├── 📁 src/
│   ├── 📁 data/
│   │   ├── 📁 ingestion/
│   │   │   ├── 📄 synthea_parser.py         ← Parse Synthea FHIR output
│   │   │   ├── 📄 ohio_parser.py            ← Parse Ohio T1DM dataset
│   │   │   └── 📄 wearable_simulator.py     ← Synthetic wearable generator
│   │   ├── 📁 preprocessing/
│   │   │   ├── 📄 feature_engineering.py   ← 89-feature computation
│   │   │   ├── 📄 clinical_indices.py      ← TIR, TAR, TBR, GMI
│   │   │   └── 📄 labeler.py              ← Adverse event labelling
│   │   └── 📁 synthetic/
│   │       ├── 📄 cgm_simulator.py         ← CGM synthetic stream
│   │       └── 📄 patient_generator.py     ← Augment Synthea patients
│   │
│   ├── 📁 twin/
│   │   ├── 📄 engine.py                    ← Twin update cycle
│   │   ├── 📄 patient_twin.py             ← PatientTwin class
│   │   ├── 📄 iob_model.py                ← Insulin-on-board PK model
│   │   └── 📄 alert_system.py             ← Tiered alert logic
│   │
│   ├── 📁 models/
│   │   ├── 📄 xgboost_model.py            ← XGBoost wrapper + SHAP
│   │   ├── 📄 lstm_model.py               ← PyTorch LSTM definition
│   │   ├── 📄 ensemble.py                 ← Weighted ensemble
│   │   └── 📁 artifacts/                  ← Saved model files
│   │       ├── 📄 xgboost_v1.pkl
│   │       └── 📄 lstm_v1.pt
│   │
│   ├── 📁 explainability/
│   │   └── 📄 shap_explainer.py           ← SHAP computation + formatting
│   │
│   ├── 📁 api/
│   │   ├── 📄 main.py                     ← FastAPI app entry point
│   │   ├── 📄 config.py                   ← Settings (env vars)
│   │   ├── 📄 database.py                 ← SQLite setup
│   │   ├── 📁 routes/
│   │   │   ├── 📄 patients.py             ← Patient CRUD + predict
│   │   │   ├── 📄 websocket.py            ← WebSocket endpoint
│   │   │   └── 📄 health.py               ← Health + metrics
│   │   ├── 📁 schemas/
│   │   │   ├── 📄 patient.py              ← Pydantic models
│   │   │   └── 📄 prediction.py
│   │   └── 📁 middleware/
│   │       └── 📄 auth.py                 ← API key middleware
│   │
│   └── 📁 dashboard/
│       ├── 📄 package.json
│       ├── 📄 vite.config.js
│       ├── 📁 public/
│       └── 📁 src/
│           ├── 📄 App.jsx
│           ├── 📄 index.css
│           ├── 📁 components/
│           │   ├── 📄 PatientSelector.jsx
│           │   ├── 📄 TwinStatusBar.jsx
│           │   ├── 📄 GlucoseTimeline.jsx
│           │   ├── 📄 AlertPanel.jsx
│           │   ├── 📄 SHAPWaterfallChart.jsx
│           │   ├── 📄 WearableMetrics.jsx
│           │   └── 📄 EHRSummary.jsx
│           └── 📁 hooks/
│               ├── 📄 useWebSocket.js
│               └── 📄 usePatientTwin.js
│
├── 📁 notebooks/
│   ├── 📄 01_eda_ehr_data.ipynb           ← EHR exploratory analysis
│   ├── 📄 02_wearable_eda.ipynb           ← CGM + sensor EDA
│   ├── 📄 03_feature_engineering.ipynb    ← Feature pipeline validation
│   ├── 📄 04_model_training.ipynb         ← XGBoost + LSTM training
│   └── 📄 05_evaluation.ipynb             ← Metrics + SHAP analysis
│
├── 📁 tests/
│   ├── 📄 test_data_pipeline.py
│   ├── 📄 test_twin_engine.py
│   ├── 📄 test_models.py
│   └── 📄 test_api.py
│
└── 📁 data/
    ├── 📁 raw/                            ← Original Synthea + Ohio output
    ├── 📁 processed/                      ← Cleaned, feature-engineered CSVs
    └── 📁 synthetic/                      ← Generated wearable streams
```

---

## 3. Deployment Architecture

```
docker-compose.yml defines 3 services:

┌─────────────────────────────────────────────────────┐
│  HOST MACHINE (localhost)                            │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  glucotwin-  │  │  glucotwin-  │  │ glucotwin │ │
│  │  dashboard   │  │  api         │  │ -db       │ │
│  │              │  │              │  │           │ │
│  │  React Vite  │  │  FastAPI     │  │ SQLite    │ │
│  │  :3000       │  │  :8000       │  │ volume    │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┘ │
│         │                 │                          │
│         │   HTTP/WS       │   SQLite (file mount)    │
│         └────────────────▶│◀────────────────────────│
│                           │                          │
└─────────────────────────────────────────────────────┘
```

---

## 4. Data Flow Sequence

```
Startup Sequence:
1. API starts → loads XGBoost + LSTM models from artifacts/
2. Demo patients pre-seeded → PatientTwin objects created
3. Wearable simulator starts → generates readings every 5 seconds (demo speed)
4. Dashboard connects via WebSocket → starts receiving twin updates

Real-time Update Cycle (every 5 min real / 5 sec demo):
  CGM Simulator ──▶ POST /api/v1/patients/{id}/update
                ──▶ PatientTwin.update(reading)
                ──▶ Feature engineering (89 features)
                ──▶ XGBoost.predict() + LSTM.predict() (parallel)
                ──▶ Ensemble → risk_score
                ──▶ Alert check → if RED: SHAP.explain()
                ──▶ State saved to SQLite
                ──▶ WebSocket broadcast to dashboard
                ──▶ Dashboard re-renders
```

---

*Architecture Document v1.0 | GlucoTwin | October 2026*
