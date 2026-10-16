# GlucoTwin Testing Infrastructure & Architecture Specification (`TEST_INFRA.md`)

## 1. Overview & Testing Strategy

GlucoTwin is a multi-modal digital twin and predictive early warning platform for Type 2 Diabetes adverse glycemic excursions. Because GlucoTwin operates in a safety-critical clinical domain with strict latency (<200ms p95), algorithmic precision, and continuous statefulness, its testing infrastructure is structured across **5 rigorous tiers**:

```text
┌────────────────────────────────────────────────────────┐
│  Tier 5: Adversarial Stress & Security Hardening       │  (Rate limits, corrupt FHIR, noise injection)
├────────────────────────────────────────────────────────┤
│  Tier 4: End-to-End Clinical Simulation Scenarios       │  (2-hour Rajesh post-meal spike, WS stream)
├────────────────────────────────────────────────────────┤
│  Tier 3: Edge Cases & Pharmacokinetic Stress Testing   │  (Hypo dropouts, zero IOB, alert fatigue)
├────────────────────────────────────────────────────────┤
│  Tier 2: Multi-Component Integration & Contracts       │  (EHR ↔ Twin ↔ XGBoost/LSTM ↔ REST/WS)
├────────────────────────────────────────────────────────┤
│  Tier 1: Unit & Component Verification                 │  (89 features, Hovorka IOB, clinical indices)
└────────────────────────────────────────────────────────┘
```

---

## 2. Test Pyramid & Tier Definitions

### Tier 1: Unit & Component Verification
- **Scope**: Isolated verification of individual functions, mathematical equations, data parsers, and statistical calculators.
- **Coverage**:
  - `SyntheaParser`: FHIR R4 extraction into 23 static features (`age`, `bmi`, `hba1c_latest`, `egfr`, `bp_sys`, medication booleans).
  - `ClinicalValidator`: Physiological ranges, eGFR contraindications (<30 mL/min on metformin), insulin consistency.
  - `FeatureEngineer`: Generation of exactly 89 features across 8 categories with zero unhandled NaNs.
  - `ClinicalIndices`: Exact mathematical calculation of TIR 24h, TAR 24h, TBR 24h, GMI %, CV % ($100 \times \sigma / \mu$).
  - `IOBModel`: Hovorka biexponential pharmacokinetic decay curve ($A=0.7$, $\tau_1 = \text{peak}/\ln(2)$, $\tau_2 = (\text{duration}-\text{peak})/\ln(2)$), reaching zero at duration.
  - `AlertSystem`: Strict tier boundaries (GREEN <0.30, YELLOW [0.30, 0.60), ORANGE [0.60, 0.80), RED $\ge$0.80).

### Tier 2: Multi-Component Integration & Contract Verification
- **Scope**: Inter-module communication and interface contracts.
- **Coverage**:
  - Ingestion pipeline (`WearableSimulator` + `CGMSimulator`) feeding into `FeatureEngineer`.
  - Feature matrix conforming to XGBoost input schema (`pandas.DataFrame` with 89 columns).
  - Temporal sequence buffer conforming to PyTorch LSTM input tensor shape `(1, 48, 3)`.
  - `EnsemblePredictor`: Patient-adaptive weighting based on `on_insulin` boolean ($0.35/0.65$ vs $0.60/0.40$).
  - FastAPI REST request/response Pydantic models (`PatientCreateRequest`, `WearableReadingRequest`, `PredictionResponse`).
  - WebSocket initial handshake (`type: "connected"`) and periodic broadcast payload schema (`type: "twin_update"`).

### Tier 3: Edge Cases, Boundary Conditions & Pharmacokinetic Stress
- **Scope**: Boundary conditions, invalid inputs, physiological anomalies, and clinical safety constraints.
- **Coverage**:
  - Insufficient history handling (<12 readings in buffer) returning `_insufficient_data_result` without crashes.
  - History buffer sequence padding (left-edge padding for sequences between 12 and 47 readings).
  - Rapid rate of change (ROC) dampening in forward trajectory forecaster ($0.90^{\text{step}}$ decay) and physiological clamping to $[40, 400]$ mg/dL.
  - Alert fatigue suppression: Suppressing duplicate ORANGE or RED alerts fired within a 30-minute window.
  - Negative or excessive elapsed time in IOB model calculation (clamping to safe upper bound and zero).
  - Sensor missingness (NaN dropouts up to 5%) and extreme glucose excursions (severe hypoglycemia < 50 mg/dL and hyperglycemia > 350 mg/dL).

### Tier 4: End-to-End Clinical Simulation Scenarios
- **Scope**: Realistic full-lifecycle patient simulation from baseline to acute event resolution.
- **Coverage**:
  - 24-hour baseline seeding (288 readings at 5-minute intervals) verifying that `is_calibrated` transitions to `True`.
  - Acute post-meal hyperglycemic excursion (24 readings / 2 hours) triggering rising glucose ($>200$ mg/dL), sympathetic HRV RMSSD drop, risk escalation into ORANGE and RED tiers.
  - Verification that RED alerts generate SHAP feature contributions with human-readable clinical labels.
  - In-memory SQLite persistence verifying complete logging of all 312 predictions and state serialization.

### Tier 5: Adversarial Stress & Security Hardening
- **Scope**: Malicious or malformed inputs, authentication bypass attempts, and concurrency stress.
- **Coverage**:
  - Unauthorized access attempts to `/api/v1/*` without `X-API-Key` or with invalid keys returning HTTP 401.
  - Duplicate patient creation attempts returning HTTP 409 Conflict.
  - Request bodies with malformed clinical schemas returning HTTP 422 Unprocessable Entity.
  - Rate limiting enforcement (>100 requests/minute returning HTTP 429).
  - Sudden WebSocket client disconnect handling and graceful cleanup without server-side leaks.

---

## 3. Test Suite Directory Layout

```text
tests/
├── conftest.py             # Shared fixtures: mock profiles, synthetic readings, engine/model mocks
├── test_data_pipeline.py   # Tier 1-3: Synthea parser, simulators, 89 features, indices, labeler
├── test_twin_engine.py     # Tier 1-3: PatientTwin state machine, IOB model, calibration, alerts
├── test_models.py          # Tier 1-3: XGBoost, LSTM, Ensemble predictor, SHAP explainability
├── test_api.py             # Tier 1-3, 5: FastAPI REST endpoints, auth, CRUD, validation, demo
├── test_websocket.py       # Tier 2, 5: Real-time WebSocket connection, 5-sec broadcasts, ping/pong
└── test_integration.py     # Tier 4: Full-lifecycle 2-hour Rajesh scenario, end-to-end integration
```

---

## 4. Test Configuration & Execution Guide

### 4.1 Pytest Execution Commands

```bash
# Run entire test suite with verbose output and coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific tier test modules
pytest tests/test_data_pipeline.py -v
pytest tests/test_twin_engine.py -v
pytest tests/test_models.py -v
pytest tests/test_api.py -v --asyncio-mode=auto
pytest tests/test_websocket.py -v
pytest tests/test_integration.py -v --asyncio-mode=auto

# Run only unit/fast tests
pytest -m "not slow" tests/

# Run with coverage threshold assertion (>80% coverage)
pytest tests/ --cov=src --cov-fail-under=80
```

### 4.2 Pytest Configuration (`pyproject.toml` or `pytest.ini`)
- Asyncio mode: `auto`
- Test discovery: `test_*.py`
- Warning filters: deprecation warnings filtered for clean terminal reporting.

---

## 5. Mathematical Formulations & Authoritative Derivation Matrix

| Metric / Mechanism | Authoritative Formula | Test Expectation |
|---|---|---|
| **Estimated Average Glucose (eAG)** | $\text{eAG} = 28.7 \times \text{HbA1c} - 46.7$ | For HbA1c 8.4%: $\text{eAG} = 194.38$ mg/dL |
| **Insulin Sensitivity Factor (ISF)** | $\text{ISF} = 1800 / \text{TDD}$ (or $50.0$ if non-insulin) | For 40 units TDD: $\text{ISF} = 45.0$ mg/dL/unit |
| **Hovorka IOB Decay** | $\text{IOB}(t) = \text{Dose} \cdot [0.7 e^{-t/\tau_1} + 0.3 e^{-t/\tau_2}]$ | At $t=0$, $\text{IOB} = \text{Dose}$; at $t \ge \text{Duration}$, $\text{IOB} = 0.0$ |
| **Glucose CV %** | $\text{CV} = (\sigma / \mu) \times 100$ | Positive percentage reflecting glycemic variability |
| **Glucose Management Indicator (GMI)** | $\text{GMI} = 3.31 + 0.02392 \times \overline{G}$ | For $\overline{G} = 150$: $\text{GMI} = 6.898\%$ |
| **Time In Range (TIR)** | $\sum \mathbb{I}(70 \le G \le 180) / N \times 100$ | Percentage between 0.0% and 100.0% |
| **Ensemble Weighted Score** | $w_{\text{XGB}} P_{\text{XGB}} + w_{\text{LSTM}} P_{\text{LSTM}}$ | Insulin: $0.35/0.65$; Non-insulin: $0.60/0.40$ |
| **Alert Fatigue Cooldown** | $\Delta t < 30 \text{ min}$ for identical tier | Suppresses alert emission |
| **Trajectory Forecast Decay** | $\text{ROC}_k = \text{ROC}_{k-1} \times 0.90$ | Prevents asymptotic runaway over 24 forecast steps |

---

## 6. Progressive Testability & Mock Architecture

To enable autonomous development across concurrent agent tracks (where data pipeline, model training, and API backend may be implemented in parallel):
1. **Dynamic Module Imports**: Test suites dynamically verify module availability and fall back to interface mock definitions where appropriate, allowing tests to run during early implementation milestones.
2. **Realistic Synthesis**: Test fixtures in `tests/conftest.py` supply realistic synthetic data adhering strictly to project schemas (23 EHR features, 89 tabular features, $(1, 48, 3)$ temporal sequence tensors, and `WearableReading` instances).
3. **In-Memory Persistence**: SQLite database fixtures use `:memory:` paths to ensure total test isolation and zero disk state leakage between test executions.
