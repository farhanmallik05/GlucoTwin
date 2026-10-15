# Vibe Assist 01 — Project Setup
## GlucoTwin Development Environment Bootstrap

> **Vibe Coding Session 01/10** | Estimated time: 2-3 hours
> Goal: Full development environment ready, repository initialized, all dependencies installed.

---

## 🎯 Session Goal

By the end of this session you will have:
- ✅ Git repository initialized and pushed to GitHub
- ✅ Python virtual environment with all dependencies
- ✅ Node.js frontend scaffold created
- ✅ Docker Compose file written
- ✅ Synthea installed and generating T2D patients
- ✅ Folder structure created exactly as per ARCHITECTURE.md

---

## 🤖 Vibe Prompt 1 — Repo & Python Setup

Copy-paste this into your AI assistant:

```
I'm starting a hackathon project called GlucoTwin — a Digital Twin for 
Type 2 Diabetes prediction. I'm a solo developer using vibe coding.

Task: Set up the Python project from scratch.

Requirements:
1. Create a requirements.txt with these packages:
   - fastapi==0.100.0
   - uvicorn[standard]==0.23.0
   - pydantic==2.0.0
   - sqlalchemy==2.0.0
   - aiosqlite==0.19.0
   - websockets==11.0
   - pandas==2.0.0
   - numpy==1.24.0
   - scikit-learn==1.3.0
   - xgboost==1.7.0
   - torch==2.0.0
   - shap==0.42.0
   - tsfresh==0.20.0
   - synthea-python (if available) or note manual Synthea install
   - pytest==7.4.0
   - httpx==0.24.0

2. Create a Python venv setup script (setup_env.sh for Mac/Linux, 
   setup_env.bat for Windows).

3. Create a .gitignore that covers Python, Node.js, and common IDE files.
   Also ignore: data/raw/, data/processed/, src/models/artifacts/*.pkl,
   src/models/artifacts/*.pt (large binary files)

4. Create a .env.example file with:
   API_KEY=your-secret-key-here
   DEMO_MODE=true
   TWIN_UPDATE_INTERVAL_SECONDS=5
   LOG_LEVEL=INFO
   DATABASE_URL=sqlite:///./glucotwin.db

Output all files with full content.
```

---

## 🤖 Vibe Prompt 2 — Folder Structure Creation

```
Create a shell script (create_structure.sh) and also a Python script 
(create_structure.py) that creates the full GlucoTwin folder structure:

GlucoTwin/
├── docs/
│   └── vibe-assist/
├── src/
│   ├── data/
│   │   ├── ingestion/
│   │   ├── preprocessing/
│   │   └── synthetic/
│   ├── twin/
│   ├── models/
│   │   └── artifacts/
│   ├── explainability/
│   ├── api/
│   │   ├── routes/
│   │   ├── schemas/
│   │   └── middleware/
│   └── dashboard/
│       └── src/
│           ├── components/
│           └── hooks/
├── notebooks/
├── tests/
└── data/
    ├── raw/
    ├── processed/
    └── synthetic/

Each directory should have an empty __init__.py (for Python dirs) or 
a .gitkeep (for data dirs). Also create a placeholder README in each 
major directory explaining its purpose.
```

---

## 🤖 Vibe Prompt 3 — Docker Compose Setup

```
Create a docker-compose.yml for the GlucoTwin project with 3 services:

1. glucotwin-api:
   - Build from ./src/api/Dockerfile
   - Port: 8000:8000
   - Environment variables from .env
   - Volumes: ./data:/app/data, ./src/models/artifacts:/app/artifacts
   - Depends on: glucotwin-db (conceptually, using SQLite so same service)

2. glucotwin-dashboard:
   - Build from ./src/dashboard/Dockerfile
   - Port: 3000:3000
   - Environment: VITE_API_URL=http://localhost:8000

3. A Makefile with commands:
   make setup    → Create venv + install deps
   make dev-api  → Run FastAPI with hot reload
   make dev-ui   → Run React dev server
   make docker   → docker-compose up --build
   make test     → pytest + frontend lint
   make demo     → Seed demo data + start all services

Also create:
- src/api/Dockerfile
- src/dashboard/Dockerfile

Both should use multi-stage builds for cleanliness.
```

---

## 🤖 Vibe Prompt 4 — Synthea Installation Script

```
I need to install and configure Synthea to generate synthetic Type 2 
Diabetes patient records. Synthea requires Java 11+.

Create:
1. A setup script (scripts/install_synthea.sh) that:
   - Downloads Synthea JAR from GitHub releases
   - Checks for Java 11+ installation
   - Creates a synthea_config/ directory
   - Writes a synthea.properties file optimized for:
     * Generating 1000 patients
     * Focus on T2D (diabetes module)
     * FHIR R4 output format
     * CSV output also enabled
   - Runs a test generation of 5 patients to verify

2. A Python script (scripts/run_synthea.py) that:
   - Invokes Synthea via subprocess
   - Generates 1000 T2D patients
   - Moves output to data/raw/synthea/
   - Prints progress and patient count

3. Document the expected Synthea FHIR output structure so I know what 
   files to expect.

Make the scripts cross-platform (Windows + Mac/Linux variants).
```

---

## ✅ Manual Steps for This Session

After running the vibe prompts above, complete these manually:

### Step 1: Create GitHub Repository

```bash
# On GitHub.com: Create new public repo named "AlgoAlchemist_[YourCollegeName]"
git init
git add .
git commit -m "feat: initial project structure and documentation"
git branch -M main
git remote add origin https://github.com/[username]/GlucoTwin_[College].git
git push -u origin main
```

### Step 2: Verify Python Environment

```bash
python --version          # Should be 3.11+
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip list | grep -E "fastapi|xgboost|torch|shap"
```

### Step 3: Verify Node.js

```bash
node --version     # Should be 18+
npm --version      # Should be 9+
```

### Step 4: Verify Java for Synthea

```bash
java --version     # Should be 11+
```

### Step 5: Verify Docker

```bash
docker --version
docker-compose --version
```

---

## 📋 Session 01 Completion Checklist

- [ ] GitHub repo created and code pushed
- [ ] `requirements.txt` complete
- [ ] `venv` created and packages installable
- [ ] `.gitignore` committed
- [ ] `.env.example` committed
- [ ] `docker-compose.yml` created
- [ ] Full folder structure created
- [ ] Synthea installed and generating patients
- [ ] `Makefile` working
- [ ] All __init__.py files created

## 🔜 Next Session

→ **02-data-pipeline.md** — Build EHR ingestion and wearable simulation

---

*GlucoTwin Vibe Assist 01 | Project Setup | Digital Twin Challenge 2026*
