# Vibe Assist 10 — Deployment & Testing
## Docker Compose, Integration Tests & Demo Preparation

> **Vibe Coding Session 10/10** | Estimated time: 4 hours
> Goal: One-command deployment, all tests passing, demo-ready state, GitHub submission ready.

---

## 🎯 Session Goal

By the end of this session you will have:
- ✅ `docker-compose up --build` brings up the full system
- ✅ All unit tests passing (pytest)
- ✅ End-to-end integration test passing
- ✅ Demo scenario rehearsed and working offline
- ✅ GitHub repo complete with all required files
- ✅ Submission checklist 100% complete

---

## 🤖 Vibe Prompt 1 — Complete Docker Compose Setup

```
Create the final docker-compose.yml for GlucoTwin.

version: '3.9'

services:

  glucotwin-api:
    build:
      context: .
      dockerfile: src/api/Dockerfile
    container_name: glucotwin-api
    ports:
      - "8000:8000"
    environment:
      - API_KEY=${API_KEY:-demo-glucotwin-2026}
      - DEMO_MODE=true
      - TWIN_UPDATE_INTERVAL_SECONDS=5
      - DATABASE_URL=sqlite:////app/data/glucotwin.db
      - LOG_LEVEL=INFO
      - MODEL_ARTIFACTS_PATH=/app/artifacts
    volumes:
      - ./data:/app/data
      - ./src/models/artifacts:/app/artifacts:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  glucotwin-dashboard:
    build:
      context: src/dashboard
      dockerfile: Dockerfile
    container_name: glucotwin-dashboard
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8000
      - VITE_WS_URL=ws://localhost:8000
      - VITE_API_KEY=${API_KEY:-demo-glucotwin-2026}
    depends_on:
      glucotwin-api:
        condition: service_healthy
    restart: unless-stopped

Create src/api/Dockerfile:
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY src/ src/
  COPY scripts/ scripts/
  RUN mkdir -p /app/data /app/artifacts
  EXPOSE 8000
  CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

Create src/dashboard/Dockerfile:
  FROM node:18-alpine AS builder
  WORKDIR /app
  COPY package*.json .
  RUN npm ci --production=false
  COPY . .
  RUN npm run build
  
  FROM nginx:alpine
  COPY --from=builder /app/dist /usr/share/nginx/html
  COPY nginx.conf /etc/nginx/conf.d/default.conf
  EXPOSE 3000
  CMD ["nginx", "-g", "daemon off;"]

Create src/dashboard/nginx.conf with:
  - Serve React app on port 3000
  - Proxy /api → http://glucotwin-api:8000/api
  - Proxy /ws → ws://glucotwin-api:8000/ws (WebSocket upgrade)
  - SPA fallback: all 404s → index.html

Create .dockerignore with:
  __pycache__, *.pyc, .git, node_modules, data/raw, venv, .env
```

---

## 🤖 Vibe Prompt 2 — Full Test Suite

```
Create a comprehensive test suite. All tests should pass with:
  pytest tests/ -v --cov=src --cov-report=term-missing

Create the following test files:

tests/test_data_pipeline.py:
  - test_synthea_parser_loads_patients()
  - test_ehr_features_correct_count()  (expect 23 features)
  - test_clinical_validator_rejects_invalid_age()
  - test_clinical_validator_rejects_inconsistent_insulin()
  - test_cgm_simulator_generates_realistic_range()  (all values 40-400)
  - test_hrv_drops_during_hypoglycaemia()
  - test_feature_engineering_outputs_89_features()
  - test_labeler_correctly_labels_events()
  - test_dataset_split_no_patient_leakage()

tests/test_twin_engine.py:
  - test_create_patient_twin()
  - test_twin_update_appends_history()
  - test_twin_calibrates_after_24h()
  - test_alert_system_generates_red_for_high_risk()
  - test_alert_fatigue_suppression()
  - test_iob_model_decays_correctly()
  - test_iob_reaches_zero_at_duration()
  - test_twin_engine_handles_multiple_patients()
  - test_persistence_save_and_load()

tests/test_models.py:
  - test_xgboost_loads_from_artifacts()
  - test_xgboost_predict_returns_probability()  (0 < p < 1)
  - test_lstm_loads_from_artifacts()
  - test_lstm_sequence_correct_shape()  ((1, 48, 3))
  - test_ensemble_returns_weighted_combination()
  - test_ensemble_weights_differ_by_insulin_status()
  - test_shap_explains_top_10_features()
  - test_shap_clinical_labels_readable()

tests/test_api.py:
  - (from vibe-assist 08, all endpoints)

tests/test_integration.py:
  - test_full_pipeline_ehr_to_prediction()
  - test_demo_scenario_produces_red_alert()
  - test_websocket_broadcasts_after_update()
  - test_docker_compose_health_check()  (optional, marks as slow)

Create conftest.py with pytest fixtures:
  - sample_ehr_features() → dict
  - sample_wearable_reading() → WearableReading
  - loaded_twin_engine() → TwinEngine (with mocked models)
  - demo_patient_twin() → PatientTwin (pre-populated 24h history)

Target: > 80% code coverage on src/
```

---

## 🤖 Vibe Prompt 3 — End-to-End Integration Test

```
Create tests/test_integration.py with a full end-to-end test
that simulates the complete GlucoTwin workflow:

async def test_full_demo_scenario():
    """
    Integration test: Full 2-hour simulation of the Rajesh scenario.
    Verifies the complete pipeline works from sensor reading to alert.
    """
    # 1. Initialize engine with real models
    engine = create_engine_with_models()
    db = TwinPersistence(":memory:")  # In-memory SQLite for tests
    
    # 2. Create demo patient
    patient_id = "INTEGRATION-001"
    ehr = sample_ehr_rajesh()
    twin = engine.create_twin(patient_id, ehr)
    
    # 3. Seed 24h baseline (normal glucose)
    for i in range(288):
        reading = WearableReading(
            patient_id=patient_id,
            timestamp=datetime.utcnow() - timedelta(minutes=(288-i)*5),
            glucose_mgdl=np.random.normal(130, 15),
            glucose_roc=np.random.normal(0, 0.3),
            hrv_rmssd_ms=np.random.normal(40, 8),
            ...
        )
        twin.update(reading)
    
    assert twin.is_calibrated, "Twin should be calibrated after 24h"
    
    # 4. Simulate post-meal spike (rising glucose + HRV drop)
    predictions = []
    for i in range(24):  # 2 hours
        glucose = 140 + i * 6 + np.random.normal(0, 5)  # Rising
        hrv = 40 - i * 0.8 + np.random.normal(0, 3)     # Dropping
        reading = WearableReading(
            glucose_mgdl=glucose,
            glucose_roc=2.0 + i * 0.1,  # Accelerating rise
            hrv_rmssd_ms=max(15, hrv),
            ...
        )
        result = twin.update(reading)
        predictions.append(result)
        await db.log_prediction(result)
    
    # 5. Assertions
    final_glucose = 140 + 24 * 6  # ~284 mg/dL
    assert twin.current_glucose > 200, f"Expected hyperglycaemia, got {twin.current_glucose}"
    
    risk_scores = [p.risk_score for p in predictions]
    assert max(risk_scores) > 0.7, f"Expected high risk peak, got {max(risk_scores)}"
    
    red_alerts = [p for p in predictions if p.risk_tier == "RED"]
    assert len(red_alerts) > 0, "Expected at least one RED alert"
    
    # Verify SHAP was generated for RED alerts
    for red_pred in red_alerts:
        assert red_pred.alert is not None
        assert red_pred.alert.shap_features is not None
    
    # Verify prediction history in DB
    history = await db.get_prediction_history(patient_id, hours=3)
    assert len(history) == 288 + 24  # Baseline + scenario
    
    print(f"\n✅ Integration test passed!")
    print(f"   Peak glucose: {twin.current_glucose:.0f} mg/dL")
    print(f"   Max risk score: {max(risk_scores):.2f}")
    print(f"   RED alerts fired: {len(red_alerts)}")
    print(f"   First RED alert at reading: {next(i for i,p in enumerate(predictions) if p.risk_tier == 'RED')}")
```

---

## 🤖 Vibe Prompt 4 — GitHub Submission Prep

```
Create the final submission preparation script: scripts/prepare_submission.py

This script does a final checklist before submission:

import os, json, subprocess
from pathlib import Path

def check_submission():
    print("🔍 GlucoTwin Submission Readiness Check")
    print("=" * 50)
    
    checks = []
    
    # 1. GitHub repo public (manual check)
    checks.append(("GitHub repo is PUBLIC", "MANUAL"))
    
    # 2. README.md complete
    readme = Path("README.md").read_text()
    required_sections = ["Team Details", "Problem Statement", "Technical Stack",
                          "Demo Video", "Repository Structure", "Open-Source License"]
    for section in required_sections:
        ok = section in readme
        checks.append((f"README: {section}", "✅" if ok else "❌"))
    
    # 3. Required docs exist
    required_docs = [
        "docs/TDR.md", "docs/PDR.md", "docs/ARCHITECTURE.md",
        "docs/HACKATHON_GUIDE.md",
        "docs/vibe-assist/01-project-setup.md",
        "docs/vibe-assist/02-data-pipeline.md",
        "docs/vibe-assist/03-ehr-simulation.md",
        "docs/vibe-assist/04-wearable-integration.md",
        "docs/vibe-assist/05-digital-twin-engine.md",
        "docs/vibe-assist/06-ml-model-training.md",
        "docs/vibe-assist/07-prediction-engine.md",
        "docs/vibe-assist/08-api-backend.md",
        "docs/vibe-assist/09-dashboard-ui.md",
        "docs/vibe-assist/10-deployment-testing.md",
    ]
    for doc in required_docs:
        ok = Path(doc).exists()
        checks.append((f"Doc: {doc}", "✅" if ok else "❌"))
    
    # 4. Model artifacts exist
    artifacts = ["src/models/artifacts/xgboost_v1.pkl",
                 "src/models/artifacts/lstm_v1.pt",
                 "src/models/artifacts/shap_explainer.pkl"]
    for artifact in artifacts:
        ok = Path(artifact).exists()
        checks.append((f"Artifact: {artifact}", "✅" if ok else "❌"))
    
    # 5. Tests pass
    result = subprocess.run(["pytest", "tests/", "-q", "--tb=no"], 
                             capture_output=True, text=True)
    ok = result.returncode == 0
    checks.append(("All tests pass", "✅" if ok else f"❌ {result.stdout.split(chr(10))[-2]}"))
    
    # 6. Docker builds
    result = subprocess.run(["docker-compose", "build", "--no-cache"], 
                             capture_output=True, text=True)
    ok = result.returncode == 0
    checks.append(("Docker Compose build succeeds", "✅" if ok else "❌"))
    
    # 7. Demo video URL in README
    has_video_url = "youtu.be" in readme or "youtube.com" in readme
    real_url = "PLACEHOLDER" not in readme
    checks.append(("Demo video URL in README", "✅" if (has_video_url and real_url) else "❌ (replace PLACEHOLDER)"))
    
    # 8. LICENSE file
    ok = Path("LICENSE").exists()
    checks.append(("LICENSE (MIT) present", "✅" if ok else "❌"))
    
    # Print results
    failed = []
    for check, status in checks:
        print(f"  {status} {check}")
        if status == "❌":
            failed.append(check)
    
    print()
    if not failed:
        print("🏆 ALL CHECKS PASSED — Ready to submit!")
    else:
        print(f"❌ {len(failed)} check(s) failed. Fix before submitting.")
        for f in failed:
            print(f"   - {f}")

if __name__ == "__main__":
    check_submission()
```

---

## 🤖 Vibe Prompt 5 — Demo Rehearsal Script

```
Create scripts/demo_rehearsal.py — a guided demo helper for the video recording 
and live presentation.

This script:
1. Starts all services
2. Seeds demo patients
3. Prints step-by-step demo instructions with timing
4. Auto-triggers events at the right time
5. Monitors that everything is working correctly

DEMO SCRIPT (for 20-minute video recording):

class DemoRehearsalGuide:
    SCRIPT = [
        {
            "time": "0:00",
            "section": "Opening Hook",
            "action": "SPEAK",
            "notes": "Open with: 'Every 4 seconds, an Indian goes blind or loses a limb due to uncontrolled diabetes...'",
            "api_action": None,
        },
        {
            "time": "1:30",
            "section": "Problem Statement",
            "action": "SHOW_SLIDE",
            "notes": "Show India T2D statistics slide. 101M patients, 1 in 6 ER visits.",
            "api_action": None,
        },
        {
            "time": "3:00",
            "section": "Architecture Overview",
            "action": "SHOW_DIAGRAM",
            "notes": "Open ARCHITECTURE.md or architecture_diagram.pdf. Walk through layers.",
            "api_action": None,
        },
        {
            "time": "5:00",
            "section": "Dashboard Introduction",
            "action": "OPEN_DASHBOARD",
            "notes": "Open http://localhost:3000. Point out: 4 patients, risk badges.",
            "api_action": "open_browser",
        },
        {
            "time": "6:00",
            "section": "Select Rajesh",
            "action": "CLICK",
            "notes": "Click DEMO-RAJESH. Explain his profile: 52yr, HbA1c 8.4%, Metformin.",
            "api_action": None,
        },
        {
            "time": "8:00",
            "section": "Normal State",
            "action": "POINT",
            "notes": "Show GREEN badge. Glucose 128 mg/dL, stable. 'This is his normal morning.'",
            "api_action": None,
        },
        {
            "time": "9:00",
            "section": "TRIGGER: Inject Post-Meal Spike",
            "action": "API_CALL",
            "notes": "Say: 'Rajesh just had a heavy lunch — biryani, no correction dose.'",
            "api_action": "inject_post_meal_spike",  # Calls API
        },
        {
            "time": "9:30",
            "section": "Watch YELLOW Appear",
            "action": "WATCH",
            "notes": "Badge changes GREEN → YELLOW. Glucose rising. Point out rate-of-change.",
            "api_action": None,
        },
        {
            "time": "11:00",
            "section": "Watch ORANGE → RED",
            "action": "WATCH",
            "notes": "Badge escalates to ORANGE then RED. Pulsing animation. Alert fires.",
            "api_action": None,
        },
        {
            "time": "12:00",
            "section": "SHAP Explanation",
            "action": "POINT",
            "notes": "Point to SHAP waterfall. Explain: 'The model tells the doctor WHY it's alerting.'",
            "api_action": None,
        },
        {
            "time": "13:30",
            "section": "Clinical Interpretation",
            "action": "EXPLAIN",
            "notes": "Walk through top 3 SHAP features. 'Rising glucose (+3.2 mg/dL/min) + HRV suppression.'",
            "api_action": None,
        },
        {
            "time": "15:00",
            "section": "Model Metrics",
            "action": "SHOW_NOTEBOOK",
            "notes": "Open notebooks/05_evaluation.ipynb. Show AUC-ROC 0.91, calibration plot.",
            "api_action": None,
        },
        {
            "time": "17:00",
            "section": "Architecture Deep Dive",
            "action": "SHOW_CODE",
            "notes": "Show PatientTwin.predict() briefly. 'Under the hood: XGBoost + LSTM ensemble.'",
            "api_action": None,
        },
        {
            "time": "18:30",
            "section": "Roadmap",
            "action": "SHOW_SLIDE",
            "notes": "Show ABDM integration slide. Real CGM device SDK. Federated learning.",
            "api_action": None,
        },
        {
            "time": "19:30",
            "section": "Closing",
            "action": "SPEAK",
            "notes": "Close with: 'With GlucoTwin, India's 101 million diabetic patients could have a guardian angel — a digital twin that never sleeps.'",
            "api_action": None,
        },
    ]
    
    def run_interactive(self):
        """Interactive rehearsal helper — prints each step and waits for ENTER."""
        for step in self.SCRIPT:
            print(f"\n⏱️  [{step['time']}] {step['section']}")
            print(f"   📋 {step['notes']}")
            if step['api_action'] == 'inject_post_meal_spike':
                input("\n   ▶️  Press ENTER to inject event...")
                self._inject_event('post_meal_spike')
                print("   ✅ Event injected! Watch the dashboard.")
            else:
                input("   ▶️  Press ENTER when ready for next step...")
    
    def _inject_event(self, event_type):
        import requests
        requests.post(
            f"http://localhost:8000/api/v1/demo/inject-event",
            params={"patient_id": "DEMO-RAJESH", "event_type": event_type},
            headers={"X-API-Key": "demo-glucotwin-2026"}
        )

if __name__ == "__main__":
    DemoRehearsalGuide().run_interactive()
```

---

## ✅ Final Pre-Submission Checklist

### Technical Verification

```bash
# Run full test suite
pytest tests/ -v --cov=src --cov-report=term
# Expected: All green, coverage > 70%

# Run submission check
python scripts/prepare_submission.py
# Expected: ALL CHECKS PASSED

# Test Docker deployment
docker-compose up --build -d
sleep 30
curl http://localhost:8000/health
# Expected: {"status": "healthy", ...}
open http://localhost:3000
# Expected: Dashboard loads with 4 demo patients

# Test demo scenario
python scripts/demo_rehearsal.py
# Complete one full rehearsal
```

### GitHub Repo Final Checks

```bash
# Ensure .gitignore is correct (no secrets committed)
git status
# Should NOT see: .env, data/raw/, *.pkl, *.pt

# Confirm all docs committed
git ls-files docs/
# Should see all 15 markdown files

# Final commit message
git add .
git commit -m "feat: complete GlucoTwin submission — all docs, models, and dashboard"
git push origin main

# Test public access (incognito browser)
# Visit: https://github.com/[username]/GlucoTwin_[College]
# Verify: All files visible, README renders correctly
```

### Video Recording Checklist

```
Before recording:
  [ ] Close all unnecessary tabs/apps
  [ ] Set screen resolution to 1920×1080
  [ ] Disable notifications (Do Not Disturb mode)
  [ ] Test microphone audio level
  [ ] Warm up Docker services (run for 5 min before recording)
  [ ] Pre-navigate to: dashboard, notebooks, architecture diagram
  [ ] Have demo script on phone/tablet (not on main screen)
  
During recording:
  [ ] Speak clearly and at a steady pace
  [ ] Pause for 1-2 seconds at key moments (for editing)
  [ ] Show the SHAP chart for at least 30 seconds (judges love this)
  [ ] State your name and college name at start
  
After recording:
  [ ] Upload to YouTube as UNLISTED (not private!)
  [ ] Copy the video URL into README.md (replace PLACEHOLDER)
  [ ] Test the YouTube URL in incognito mode
  [ ] Push README update to GitHub
```

---

## 📋 Session 10 Completion Checklist

- [ ] `docker-compose up --build` works end-to-end
- [ ] All tests pass (> 80% coverage)
- [ ] Integration test: RED alert fires during scenario
- [ ] `prepare_submission.py` shows ALL CHECKS PASSED
- [ ] Demo video recorded (20+ minutes, unlisted YouTube)
- [ ] YouTube URL in README.md
- [ ] GitHub repo pushed and publicly accessible
- [ ] Repository name: `AlgoAlchemist_[YourCollegeName]`
- [ ] Submission form filled with: name, phone, email, repo link

---

## 🏁 Congratulations!

You've completed all 10 vibe coding sessions. GlucoTwin is ready to compete.

**Your winning advantage:**
- **Only team** with SHAP explainability in the dashboard
- **Only team** with a stateful Digital Twin (vs. simple predictor)
- **India-specific** patient demographics (judges will notice)
- **Professional documentation** (TDR + PDR + 10 vibe-assist docs)
- **Demo WOW moment**: Real-time RED alert firing with clinical explanation

> *"The future of medicine is proactive, not reactive. GlucoTwin is that future."*

---

*GlucoTwin Vibe Assist 10 | Deployment & Testing | Digital Twin Challenge 2026*
*All 10 Vibe Assist Sessions Complete ✅*
