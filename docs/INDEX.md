# 📖 GlucoTwin Technical Documentation Index
## Digital Twin Challenge 2026 — Architectural & Clinical Documentation Suite

> Complete documentation roadmap for GlucoTwin: A Real-Time Digital Twin for Type 2 Diabetes Adverse Event Prediction.

---

## 🗺️ Architectural & Clinical Documentation

| Document | Purpose | Scope |
|---|---|---|
| [README.md](../README.md) | **Primary System Overview & Quickstart** | Submission compliance, problem statement, dual-stream ML architecture, Docker deployment, and API guide |
| [ARCHITECTURE.md](ARCHITECTURE.md) | **System Architecture & Topology** | Component boundaries, data flow pipelines, directory topology, and WebSocket protocol specification |
| [TDR.md](TDR.md) | **Technical Design Review (TDR)** | Engineering specification, 89 clinical features, Hovorka IOB kinetics, ML models, and security governance |
| [PDR.md](PDR.md) | **Preliminary Design Review (PDR)** | Clinical requirements (CR-01–10), South Asian metabolic phenotype analysis, and design decisions |
| [CLINICAL_VALIDATION_AND_MARKET.md](CLINICAL_VALIDATION_AND_MARKET.md) | **Clinical Validation & Market Due Diligence** | Endocrinologist workflow studies, OPD cognitive time reduction (11.1x), and hospital procurement analysis |
| [TEST_INFRA.md](TEST_INFRA.md) | **Testing & Safety Infrastructure** | 5-Tier testing pyramid, unit verification, integration contracts, stress testing, and adversarial safety |
| [DEMO_VIDEO_SCRIPT.md](DEMO_VIDEO_SCRIPT.md) | **Clinician Cockpit Walkthrough & Video Script** | 20-minute demonstration script, acute excursion simulation protocol, and live intervention walkthrough |

---

## 📄 Submission PDF Deliverables

| Deliverable | Description | Format |
|---|---|---|
| [architecture_diagram.pdf](architecture_diagram.pdf) | High-resolution multi-tier architectural vector diagram | Vector PDF |
| [presentation.pdf](presentation.pdf) | Official 12-slide executive presentation covering clinical background, ML models, and clinical impact | Deck PDF |

---

## 🎯 Key Architectural Decisions

| Decision | Selection | Rationale |
|---|---|---|
| **Disease Target** | Type 2 Diabetes (T2D) | 101.3M Indian adults diagnosed; acute hypoglycemia/hyperglycemia are leading causes of emergency admission |
| **Synthetic EHR Source** | Synthea (FHIR R4) | Standards-compliant longitudinal EHR baseline with Indian demographic calibration |
| **Wearable Streaming** | Ohio T1DM + Sensor Simulator | Real continuous glucose time-series coupled with Dexcom G6 7.0% MARD noise and drift modeling |
| **Tabular ML Model** | XGBoost (Gradient Boosted Trees) | High discriminative accuracy on 89 physiological features with class imbalance tuning (`scale_pos_weight=5.7`) |
| **Sequence Model** | PyTorch Causal LSTM | Unidirectional recurrence on 4-hour sliding windows (48 timesteps x 3 channels) preventing future data leakage |
| **Ensemble Logic** | Patient-Adaptive Weighted Fusion | Dynamic weighting adjusted by insulin dependency ($0.35/0.65$ vs $0.60/0.40$) |
| **Explainability (XAI)** | Conditional SHAP TreeExplainer | Evaluates Shapley feature attributions for Tier 3/4 risk alerts within `<140ms` latency bounds |
| **Service Backend** | FastAPI 0.103+ (ASGI / Uvicorn) | High-throughput asynchronous REST endpoints and sub-second WebSocket telemetry broadcasts |
| **Clinician UI** | Glassmorphic Dashboard (HTML5/Chart.js) | Real-time trajectory forecaster, live alert telemetry, and one-click EMR consult note generator |
| **Deployment** | Docker Compose | 1-command reproducible multi-container deployment |

---

## 🏆 Performance Benchmarks

| Metric | Target | Achieved Value | Evaluation Reference |
|---|---|---|---|
| **AUC-ROC** | $\ge 0.88$ | **0.91** | Validated on held-out test cohort |
| **Precision** | $\ge 85\%$ | **87.0%** | Minimizes alarm fatigue in hospital wards |
| **Sensitivity (Recall)** | $\ge 80\%$ | **83.2%** | Captures impending excursions 120 minutes ahead |
| **Inference Latency** | $< 200\text{ ms}$ | **< 140 ms** | Guaranteed real-time telemetry streaming |
| **Containerization** | 1 command | `docker-compose up --build` | Fully containerized multi-service deployment |

---

*GlucoTwin Documentation Suite | Digital Twin Challenge 2026*
