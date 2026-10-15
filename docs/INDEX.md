# 📖 GlucoTwin Documentation Index
## Digital Twin Challenge 2026 — Complete Documentation Suite

> This index maps every document in the project to its purpose and the order to read them.

---

## 🗺️ Quick Navigation

| Document | Purpose | Priority |
|---|---|---|
| [README.md](../README.md) | GitHub submission page | **Read First** |
| [HACKATHON_GUIDE.md](HACKATHON_GUIDE.md) | Winning strategy for solo developers | **Read Before Building** |
| [PDR.md](PDR.md) | Problem analysis, clinical requirements, design decisions | Before coding |
| [TDR.md](TDR.md) | Full technical specification and architecture | During coding |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System components and directory structure | Reference |

---

## 🤖 Vibe Coding Assist Sessions (In Order)

| Session | Document | Focus Area | Day |
|---|---|---|---|
| 01 | [01-project-setup.md](vibe-assist/01-project-setup.md) | Environment, GitHub, Synthea | Day 1 |
| 02 | [02-data-pipeline.md](vibe-assist/02-data-pipeline.md) | EHR ingestion, wearable sim, dataset | Days 2-3 |
| 03 | [03-ehr-simulation.md](vibe-assist/03-ehr-simulation.md) | India T2D demographics, clinical QA | Day 3 |
| 04 | [04-wearable-integration.md](vibe-assist/04-wearable-integration.md) | CGM noise, HRV sim, real-time streams | Day 4 |
| 05 | [05-digital-twin-engine.md](vibe-assist/05-digital-twin-engine.md) | PatientTwin class, IOB model, alerts | Day 5 |
| 06 | [06-ml-model-training.md](vibe-assist/06-ml-model-training.md) | XGBoost + LSTM + SHAP training | Days 5-6 |
| 07 | [07-prediction-engine.md](vibe-assist/07-prediction-engine.md) | Twin.predict(), latency profiling | Day 7 |
| 08 | [08-api-backend.md](vibe-assist/08-api-backend.md) | FastAPI REST + WebSocket | Days 8-9 |
| 09 | [09-dashboard-ui.md](vibe-assist/09-dashboard-ui.md) | React charts, real-time updates | Days 10-11 |
| 10 | [10-deployment-testing.md](vibe-assist/10-deployment-testing.md) | Docker, tests, submission | Days 12-14 |

---

## 📊 Document Summary

### HACKATHON_GUIDE.md
The strategic playbook. Contains:
- Challenge scoring framework analysis
- Why Type 2 Diabetes is the winning condition choice
- Solo vibe-coding workflow for 14 days
- Judge psychology and how to win each panellist
- Demo video script (20-minute breakdown)
- Complete submission checklist

### PDR.md (Preliminary Design Review)
The "why" document. Contains:
- India T2D clinical context (101M patients, statistics)
- Root cause analysis of adverse events
- Clinical requirements (CR-01 through CR-10)
- Alternative approaches evaluated and rejected
- Data strategy and sourcing plan
- ML approach selection matrix
- Feasibility assessment for solo 14-day timeline
- Go/No-Go decision gates

### TDR.md (Technical Design Review)
The "how" document. Contains:
- Full system architecture diagram (ASCII art)
- Complete EHR and wearable data schemas
- All 89 feature specifications with clinical rationale
- PatientTwin class design
- XGBoost + LSTM architecture specifications
- Ensemble strategy with adaptive weights
- SHAP explanation format
- All API endpoints with schemas
- Alert system design (4-tier)
- Security and DPDP Act compliance
- Performance specifications
- Test pyramid and test scenarios
- Risk register

### ARCHITECTURE.md
The "what" document. Contains:
- Full system topology diagram
- Complete directory and file structure
- Deployment architecture (Docker Compose)
- Data flow sequence diagrams

---

## 🎯 Key Technical Decisions (Quick Reference)

| Decision | Choice | Why |
|---|---|---|
| Disease Focus | Type 2 Diabetes | Best datasets, highest India burden |
| EHR Source | Synthea | Free, FHIR-native, T2D module |
| Wearable Source | Ohio T1DM + Custom Sim | Real CGM data + augmentation |
| Tabular ML | XGBoost | Best for clinical tabular data |
| Temporal ML | LSTM (PyTorch) | Best for CGM time-series |
| Ensemble | Adaptive weighted | Different patients need different weights |
| Explainability | SHAP | Best XGBoost integration, clinical labels |
| Backend | FastAPI | Async, auto-docs, WebSocket native |
| Frontend | React + Recharts | Mature charting library |
| Deployment | Docker Compose | Reproducible for judge evaluation |

---

## 🏆 Success Metrics

| Metric | Target | How to Measure |
|---|---|---|
| Model AUC-ROC | ≥ 0.88 | `notebooks/05_evaluation.ipynb` |
| Model Precision | ≥ 85% | `notebooks/05_evaluation.ipynb` |
| API Latency | < 200ms (p95) | `scripts/profile_prediction.py` |
| Dashboard FPS | ≥ 30 | Chrome DevTools |
| Docker Deploy | 1 command | `docker-compose up --build` |
| Demo Video | ≥ 20 min | YouTube upload |

---

*Documentation Index v1.0 | GlucoTwin | Digital Twin Challenge 2026*
