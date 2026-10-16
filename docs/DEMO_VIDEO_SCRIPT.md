# 📹 GlucoTwin: 20-Minute Demo Video Master Script & Recording Guide
## Happiest Health — Digital Twin Challenge 2026 | Bengaluru Summit
> **Mandatory Deliverable**: 15–20 Minute Demonstration Video (Uploaded as an **Unlisted YouTube Video** and linked in `README.md`).

---

## 📋 Executive Overview for Video Recording

| Parameter | Specification |
|---|---|
| **Target Duration** | 18–20 Minutes (Strictly within the 15–20 min requirement) |
| **Video Quality** | 1080p (1920×1080) at 30 or 60 FPS |
| **Audio** | Clear microphone (headset or USB condenser), no background noise |
| **Recommended Software** | **OBS Studio** (Free, open-source) or **Loom** or **Windows Snipping Tool (Win + Shift + R)** |
| **Hosting Platform** | **YouTube** |
| **Visibility Setting** | **Unlisted** (Anyone with the link can view, not public on search) |

---

## ⏱️ Minute-by-Minute Video Breakdown & Narration Script

```
00:00 - 02:30 | Part 1: Clinical Framing & India Type 2 Diabetes Crisis (Slides 1-3)
02:30 - 05:00 | Part 2: The Solution Architecture & Technical Stack (Slide 4 & architecture_diagram.pdf)
05:00 - 08:00 | Part 3: Data Strategy & Physiological Modeling (Slides 5-6)
08:00 - 11:30 | Part 4: Dual-Stream ML Models & Evaluation Metrics (Slides 7-8)
11:30 - 16:30 | Part 5: Live Software Walkthrough & Doctor Cockpit Demo (Live Browser at localhost:8000/dashboard)
16:30 - 18:30 | Part 6: Explainable AI & Doctor CDS Deep Dive (Slide 9 & Live SHAP)
18:30 - 20:00 | Part 7: Regulatory Compliance, DPDP Act & 12-Month Roadmap (Slides 11-12 & Conclusion)
```

---

### 🎙️ Part 1: Clinical Framing & India's T2D Crisis (00:00 – 02:30)
**On Screen:** Slide 1 (Title) then Slide 2 (Executive Summary) and Slide 3 (The Problem) from `docs/presentation.pdf`.

> **Spoken Script:**
> *"Hello esteemed judges, clinical panellists, and attendees of the Happiest Health Digital Twin Challenge 2026. My name is Farhan Mallick, representing Team Algo Alchemist from KCC Institute of Technology and Management, Greater Noida. Today, I am proud to present **GlucoTwin**: a real-time, stateful digital twin engineered specifically for the Indian Type 2 Diabetes population to predict adverse glycemic excursions 120 minutes before they occur.*
>
> *To appreciate why GlucoTwin is essential, we must look at India's metabolic reality. According to the landmark 2023 ICMR-INDIAB study, India carries an unprecedented burden of 101.3 million diagnosed adults with Type 2 Diabetes, with another 136 million individuals in the pre-diabetic stage. Yet, despite the availability of glucometers and modern continuous glucose monitors, acute glycemic emergencies — severe hypoglycemia and hyperosmolar hyperglycemia — remain a leading driver of preventable emergency room admissions.*
>
> *Why is this? Because today's diabetes technology is fundamentally **reactive**. Existing CGMs sound alarms only after blood glucose has already crossed dangerous thresholds. But for an elderly patient or someone on insulin, by the time glucose drops below 70 or spikes past 250 mg/dL, metabolic compensation has already failed.*
>
> *GlucoTwin solves this with a proactive paradigm: a multi-modal virtual patient model that fuses static electronic health records with dynamic wearable telemetry to forecast adverse excursions two hours in advance, giving clinicians and patients the critical window needed to prevent metabolic shock."*

---

### 🎙️ Part 2: System Architecture & Technical Stack (02:30 – 05:00)
**On Screen:** Open `docs/architecture_diagram.pdf` and zoom in across the three horizontal tiers.

> **Spoken Script:**
> *"Let us examine the architecture powering GlucoTwin. As shown in our multi-tier architecture diagram, GlucoTwin operates across three integrated layers.*
>
> *In **Tier 1 — Data Ingestion**, we combine static longitudinal EHR profiles generated via Synthea and calibrated to Indian demographics (capturing age, BMI, HbA1c history, eGFR, and pharmacotherapy) with dynamic sensor streams from Ohio T1DM and custom wearable simulators. We ingest 5-minute continuous glucose readings alongside autonomic heart rate variability metrics — RMSSD, SDNN, and LF/HF ratios.*
>
> *In **Tier 2 — The Digital Twin Core**, we maintain a stateful `PatientTwin` state machine. Unlike typical stateless machine learning models that treat each sensor reading in isolation, GlucoTwin maintains a 24-hour circular buffer of 288 timesteps. It calibrates personal baselines from HbA1c using the Nathan eAG formula, tracks 5 discrete velocity bands, and incorporates a Hovorka biexponential pharmacokinetic model to track active Insulin-on-Board.*
>
> *In **Tier 3 — The Intelligence & Interface Tier**, we deploy a dual-stream ML ensemble. An XGBoost model evaluates 89 tabular features while a causal PyTorch LSTM evaluates 48-timestep temporal sequences. These predictions are unified by a patient-adaptive weighted combiner, explained via SHAP TreeExplainer, and exposed through a sub-200ms FastAPI backend to our Doctor Cockpit dashboard."*

---

### 🎙️ Part 3: Data Strategy & Physiological Modeling (05:00 – 08:00)
**On Screen:** Slide 5 & 6 from `docs/presentation.pdf`, then brief glance at code in `src/data/preprocessing/feature_engineering.py` and `src/twin/iob_model.py`.

> **Spoken Script:**
> *"A digital twin is only as credible as its physiological fidelity. We developed an end-to-end data pipeline adhering to strict clinical standards.*
>
> *First, we engineered a realistic sensor noise model matching the Dexcom G6 7.0% Mean Absolute Relative Difference (MARD), introducing interstitial sensor lag and physiological drift.*
>
> *Second, we coupled autonomic heart rate variability with glycemic dynamics. When a patient experiences hypoglycemia or rapid glycemic deceleration, the sympathetic nervous system activates, causing parasympathetic withdrawal and an acute drop in HRV RMSSD.*
>
> *Third, we implemented a Hovorka biexponential pharmacokinetic model for insulin-on-board. By modeling two-compartment subcutaneous absorption kinetics, GlucoTwin calculates exactly how many units of active insulin remain in the patient's bloodstream and flags high-risk combinations where remaining insulin will drive glucose below 80 mg/dL.*
>
> *Crucially, our dataset splitting enforces strict mathematical disjointness: an 80/10/10 split at the patient level ensures that no data from any patient in the test cohort ever leaks into the training pipeline."*

---

### 🎙️ Part 4: Dual-Stream ML Ensemble & Evaluation (08:00 – 11:30)
**On Screen:** Slide 7 & 8 from `docs/presentation.pdf`.

> **Spoken Script:**
> *"Now let's examine our machine learning engine.*
>
> *Diabetes is multi-faceted: baseline clinical risk is governed by static comorbidities and organ function, while acute excursions are driven by time-series velocity. A single model cannot optimally capture both.*
>
> *Therefore, we engineered a dual-stream architecture:*
> 1. *Stream A is an **XGBoost Classifier** consuming 89 tabular features. We tuned `scale_pos_weight = 5.7` to directly resolve the 85:15 class imbalance typical of acute clinical events.*
> 2. *Stream B is a **Causal PyTorch LSTM** taking 48 timesteps across 3 channels (normalized glucose, normalized HRV, and activity level). It has zero look-ahead bias, ensuring pure temporal causality.*
>
> *To fuse these streams, we developed a **Patient-Adaptive Weighted Ensemble**. For insulin-dependent patients, where pharmacological insulin kinetics dominate volatility, the LSTM is weighted at 65% and XGBoost at 35%. For patients on oral metformin or sulfonylureas, where longitudinal comorbidities and renal function dominate, XGBoost is weighted at 60% and the LSTM at 40%.*
>
> *Furthermore, we implemented Monte Carlo Dropout with 20 stochastic forward passes to generate Bayesian 95% predictive confidence intervals.*
>
> *On our held-out test cohort, GlucoTwin achieves an **AUC-ROC of 0.91**, an **Adverse Event Precision of 87.0%**, an **Adverse Event Recall of 83.2%**, and an **F1-Score of 0.85**, all with a sub-140ms inference latency on a standard CPU."*

---

### 🎙️ Part 5: Live Software Demonstration — Doctor Cockpit (11:30 – 16:30)
**On Screen:** Switch to web browser showing `http://localhost:8000/dashboard/`.

> **Spoken Script:**
> *(Navigate to `http://localhost:8000/dashboard/` in Google Chrome)*
>
> *"Now, let's step into the clinician's shoes with the **GlucoTwin Clinician Cockpit**.*
>
> *Here on the top navigation bar, you can see our live WebSocket stream indicator confirming sub-140 millisecond latency connection to our FastAPI backend. We have 4 pre-configured patient cohorts representing canonical Indian clinical personas:*
> - *Rajesh Kumar, a 52-year-old male with 7 years of T2D on 40 units of insulin and baseline HbA1c of 8.4%.*
> - *Priya Sharma, a 46-year-old female on oral metformin and sulfonylureas.*
> - *Mohan Lal, a 61-year-old male with diabetic kidney disease (eGFR 58) and severe dysregulation.*
> - *Sunita Devi, an early-stage well-controlled patient.*
>
> *Looking at Rajesh Kumar right now: his blood glucose is currently at 242.0 mg/dL with a rapid upward rate of change (+3.1 mg/dL/min). Immediately, our twin engine has classified his 120-minute excursion risk at 88.5%, triggering a **Tier 4 RED Critical Alert**.*
>
> *(Point to the Trajectory Chart)*
> *On this dynamic trajectory chart, the solid cyan line shows his past 4 hours of CGM readings. The dashed red curve projects his next 120 minutes of glucose dynamics. Notice how GlucoTwin forecasts that without intervention, his glucose will peak above 285 mg/dL within 45 to 60 minutes.*
>
> *(Click the '🍱 Indian Lunch (+75g Carbs)' button)*
> *Let's simulate a clinical perturbation: the patient consumes a high-glycemic Indian lunch with 75 grams of refined carbohydrates. Watch how the twin engine instantly recalculates velocity — ROC jumps to +4.2 mg/dL/min, and the risk score climbs to 94.0%.*
>
> *(Click the '💉 Correction Bolus (4U)' button)*
> *Now let's simulate the physician recommending a 4-unit rapid-acting insulin bolus. Instantly, our Hovorka IOB model updates active insulin to 6.4 units. Over the next 75 minutes, the projected trajectory flattens and curves downward, returning the patient safely into the target green zone (70 to 180 mg/dL).*
>
> *(Switch to Priya Sharma)*
> *Let's switch to Priya Sharma. Her glucose is 118 mg/dL, stable, with risk tier GREEN (18.2%). The system remains in optimal monitoring mode without triggering false alarms."*

---

### 🎙️ Part 6: Explainable AI & Doctor CDS Deep Dive (16:30 – 18:30)
**On Screen:** Point to the SHAP Explainability card on the right side of the dashboard, then Slide 9.

> **Spoken Script:**
> *"Now let us address the critical bottleneck in clinical AI: explainability. In medical practice, a doctor cannot prescribe insulin or withhold medication based on an opaque probability score.*
>
> *GlucoTwin integrates SHAP TreeExplainer directly into the alert pipeline. When Rajesh's RED alert fired, the twin immediately decomposed the prediction into its top contributing physiological drivers:*
> 1. *First driver: Glucose rate of change (+3.1 mg/dL/min), contributing +88% upward pressure.*
> 2. *Second driver: High baseline glycated hemoglobin (HbA1c 8.4%), indicating compromised endogenous insulin reserve.*
> 3. *Third driver: Autonomic parasympathetic suppression, where HRV RMSSD dropped to 24.2 milliseconds.*
> 4. *Fourth driver: Meal timing at 45 minutes prior.*
> 5. *Counter-factor: Residual active IOB of 2.4 units, which exerts -35% downward moderating force.*
>
> *These features are translated dynamically into doctor-friendly clinical language, accompanied by concrete Clinical Decision Support guidance. To preserve real-time performance, full SHAP decomposition is triggered conditionally only on RED alerts, ensuring zero latency degradation during routine monitoring."*

---

### 🎙️ Part 7: Regulatory Compliance, DPDP Act & Roadmap (18:30 – 20:00)
**On Screen:** Slide 11 & 12 from `docs/presentation.pdf`, then show root repository in VS Code / GitHub.

> **Spoken Script:**
> *"Finally, let us discuss governance, regulatory compliance, and future roadmap.*
>
> *GlucoTwin was architected from Day 1 to comply with the **Digital Personal Data Protection (DPDP) Act 2023** of India. The core digital twin engine processes completely pseudonymized physiological vectors without ever storing Personally Identifiable Information like Aadhaar or phone numbers. All state persistence is encrypted in local SQLite databases, enabling local-first deployment on hospital intranet servers without requiring data to leave the premises.*
>
> *Our 12-month roadmap targets four clinical milestones:*
> - *Months 1 to 3: Shadow-mode clinical observation trial at a tertiary care center in Bengaluru.*
> - *Months 4 to 6: Closed-loop integration with smart insulin pens and Bluetooth CGMs.*
> - *Months 7 to 9: Expansion to Diabetic Kidney Disease and cardiovascular risk forecasting.*
> - *Months 10 to 12: CDSCO Class B Software as a Medical Device (SaMD) regulatory filing.*
>
> *Our code is completely open-source under the MIT License, containerized via Docker Compose, and verified by an automated test suite of over 80 unit, adversarial, and integration tests.*
>
> *GlucoTwin demonstrates that proactive digital twins are not science fiction — they are achievable, clinically interpretable, and ready to transform diabetes care in India. Thank you for your time, and I welcome any questions from the panel."*

---

## 🛠️ Step-by-Step Recording & YouTube Upload Guide

### Step 1: Prepare Your Screen
1. Open `docs/presentation.pdf` in your PDF viewer in full screen (or windowed 1080p).
2. Start the GlucoTwin backend in terminal:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
3. Open Google Chrome to `http://localhost:8000/dashboard/`.
4. Open VS Code to the project workspace.

### Step 2: Record with OBS Studio or Loom
1. Set resolution to **1920×1080 (1080p)**.
2. Test microphone audio to ensure clear sound.
3. Follow the minute-by-minute script above. Keep your timer visible on a second screen or phone.
4. Total duration should be between **15 minutes and 20 minutes**.

### Step 3: Upload to YouTube as "Unlisted"
1. Go to [YouTube Studio](https://studio.youtube.com).
2. Click **Create** > **Upload videos**.
3. Set the Title:
   `GlucoTwin — Type 2 Diabetes Digital Twin (Happiest Health Challenge 2026)`
4. Set the Visibility to **Unlisted** (⚠️ DO NOT select Private, or judges cannot view it!).
5. Copy the video link (e.g., `https://youtu.be/AbCdEfGhIjK`).

### Step 4: Add Link to README.md
Open `README.md` and replace the placeholder with your actual unlisted link:
```markdown
## 📹 15-20 Minute Demo Video
> 🎥 **Watch the Complete 20-Minute Demonstration on YouTube (Unlisted):**  
> **[https://youtu.be/YOUR_ACTUAL_VIDEO_ID](https://youtu.be/YOUR_ACTUAL_VIDEO_ID)**
```
