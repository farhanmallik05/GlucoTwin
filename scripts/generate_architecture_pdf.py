"""
scripts/generate_architecture_pdf.py
Generates a publication-grade, vector-quality PDF of the GlucoTwin Multi-Tier System Architecture.
Saved directly to docs/architecture_diagram.pdf for hackathon submission compliance.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle

def create_architecture_diagram():
    fig = plt.figure(figsize=(16, 11), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Dark-mode professional theme
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    # Title & Subtitle
    ax.text(8.0, 10.4, "GLUCOTWIN: SYSTEM ARCHITECTURE & DATA FLOW", 
            fontsize=20, fontweight="bold", color="#F8FAFC", ha="center")
    ax.text(8.0, 10.05, "Type 2 Diabetes Digital Twin for 120-Minute Adverse Glycemic Excursion Prediction | DTC 2026", 
            fontsize=11, color="#94A3B8", ha="center", style="italic")

    # Helper function for drawing rounded tier cards
    def draw_box(x, y, w, h, title, subtitle, color, border_color):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor=border_color, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.35, title, fontsize=10, fontweight="bold", 
                color="#FFFFFF", ha="center", va="center", zorder=3)
        ax.text(x + w / 2, y + h / 2 - 0.2, subtitle, fontsize=8, color="#CBD5E1", 
                ha="center", va="center", zorder=3, multialignment="center")

    def draw_container(x, y, w, h, label, bg_color="#1E293B", border="#334155"):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                             facecolor=bg_color, edgecolor=border, linewidth=1.2, linestyle="--", zorder=1)
        ax.add_patch(box)
        ax.text(x + 0.3, y + h - 0.3, label, fontsize=11, fontweight="bold", 
                color="#38BDF8", ha="left", va="center", zorder=2)

    # 1. DATA INGESTION TIER
    draw_container(0.6, 7.0, 14.8, 2.5, "TIER 1: MULTI-MODAL DATA INGESTION & SENSOR SIMULATION")
    draw_box(1.0, 7.3, 3.2, 1.8, "EHR Demographics", "Synthea / FHIR R4\n101M India T2D Profile\nHbA1c, eGFR, BMI, Meds", "#1E3A8A", "#60A5FA")
    draw_box(4.6, 7.3, 3.2, 1.8, "Continuous Glucose (CGM)", "Ohio T1DM + AR(1) Sim\n5-Min Sampling [40-400 mg/dL]\nMARD 7% Sensor Noise", "#1E3A8A", "#60A5FA")
    draw_box(8.2, 7.3, 3.2, 1.8, "Wearable Autonomic Biometrics", "HRV (RMSSD, SDNN, LF/HF)\nSteps & Metabolic Activity\nParasympathetic Suppression", "#1E3A8A", "#60A5FA")
    draw_box(11.8, 7.3, 3.2, 1.8, "Data Ingestion Streamer", "WearableStreamManager\nData Quality & Gap Padding\nClinicalValidator Filter", "#1E3A8A", "#60A5FA")

    # 2. FEATURE ENGINEERING & DIGITAL TWIN CORE
    draw_container(0.6, 4.0, 14.8, 2.6, "TIER 2: FEATURE PIPELINE & STATEFUL DIGITAL TWIN ENGINE")
    draw_box(1.0, 4.3, 3.2, 1.9, "89-Feature Engineering", "Static EHR (23) + CGM Stats (20)\nClinical Indices (12: TIR, MAGE)\nHRV Autonomic Coupling (12)", "#065F46", "#34D399")
    draw_box(4.6, 4.3, 3.2, 1.9, "PatientTwin State Machine", "Circular Buffers (288 steps / 24h)\neAG Baseline = 28.7 * HbA1c - 46.7\nVelocity Bands (5 Trend States)", "#065F46", "#34D399")
    draw_box(8.2, 4.3, 3.2, 1.9, "Pharmacokinetic IOB Model", "Hovorka Biexponential Model\nActive Insulin-on-Board\nRule of 1800 ISF Sensitivity", "#065F46", "#34D399")
    draw_box(11.8, 4.3, 3.2, 1.9, "Multi-Patient TwinEngine", "Concurrent Twin Registry\nAsync State Persistence (SQLite)\nFatigue-Suppressed Alerts", "#065F46", "#34D399")

    # 3. ML INFERENCE & EXPLAINABILITY
    draw_container(0.6, 1.0, 14.8, 2.6, "TIER 3: DUAL-STREAM ML ENSEMBLE, EXPLAINABILITY & DOCTOR COCKPIT")
    draw_box(1.0, 1.3, 3.2, 1.9, "XGBoost Tabular Predictor", "89 Engineered Features\nscale_pos_weight = 5.7\nTree Hist Method (AUC 0.91)", "#701A75", "#F472B6")
    draw_box(4.6, 1.3, 3.2, 1.9, "PyTorch Temporal LSTM", "48-Timestep Causal Window\n(Batch, 48, 3 Channels)\nMonte Carlo Dropout 95% CI", "#701A75", "#F472B6")
    draw_box(8.2, 1.3, 3.2, 1.9, "Adaptive Ensemble & SHAP", "Insulin: 35% XGB / 65% LSTM\nNon-Insulin: 60% XGB / 40% LSTM\nTop 10 SHAP Explanations", "#701A75", "#F472B6")
    draw_box(11.8, 1.3, 3.2, 1.9, "FastAPI & Clinician Cockpit", "Sub-200ms REST Endpoints\nLive WebSocket Stream (/ws)\nDoctor Twin Cockpit UI", "#701A75", "#F472B6")

    # Connection Arrows (Vertical & Horizontal flows)
    arrow_props = dict(arrowstyle="->", color="#38BDF8", lw=2, mutation_scale=15)
    
    # Inter-tier connectors
    ax.annotate("", xy=(2.6, 6.2), xytext=(2.6, 7.3), arrowprops=arrow_props)
    ax.annotate("", xy=(6.2, 6.2), xytext=(6.2, 7.3), arrowprops=arrow_props)
    ax.annotate("", xy=(9.8, 6.2), xytext=(9.8, 7.3), arrowprops=arrow_props)
    ax.annotate("", xy=(13.4, 6.2), xytext=(13.4, 7.3), arrowprops=arrow_props)

    ax.annotate("", xy=(2.6, 3.2), xytext=(2.6, 4.3), arrowprops=arrow_props)
    ax.annotate("", xy=(6.2, 3.2), xytext=(6.2, 4.3), arrowprops=arrow_props)
    ax.annotate("", xy=(9.8, 3.2), xytext=(9.8, 4.3), arrowprops=arrow_props)
    ax.annotate("", xy=(13.4, 3.2), xytext=(13.4, 4.3), arrowprops=arrow_props)

    # Horizontal connectors inside ML tier
    h_arrow = dict(arrowstyle="->", color="#F472B6", lw=1.5, mutation_scale=12)
    ax.annotate("", xy=(4.6, 2.25), xytext=(4.2, 2.25), arrowprops=h_arrow)
    ax.annotate("", xy=(8.2, 2.25), xytext=(7.8, 2.25), arrowprops=h_arrow)
    ax.annotate("", xy=(11.8, 2.25), xytext=(11.4, 2.25), arrowprops=h_arrow)

    # Key Performance Callout Box at bottom
    ax.text(8.0, 0.45, "Key Metrics: AUC-ROC: 0.91 | Adverse Event Precision: 87% | Latency: < 200ms | 120-Minute Early Intervention Window | DPDP Act 2023 Compliant",
            fontsize=9.5, fontweight="bold", color="#34D399", ha="center")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "architecture_diagram.pdf")
    
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    create_architecture_diagram()
