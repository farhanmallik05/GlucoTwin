# Vibe Assist 06 — ML Model Training
## XGBoost + LSTM Ensemble Training & Evaluation

> **Vibe Coding Session 06/10** | Estimated time: 5-6 hours
> Goal: Train, evaluate, and save the ML ensemble with AUC-ROC ≥ 0.88.

---

## 🎯 Session Goal

By the end of this session you will have:
- ✅ XGBoost model trained on 89 EHR + CGM features, AUC ≥ 0.87
- ✅ LSTM model trained on 4h CGM + HRV + activity sequences, AUC ≥ 0.85
- ✅ Ensemble achieving AUC ≥ 0.90 on held-out test set
- ✅ SHAP explainer fitted and saved
- ✅ Model evaluation notebook (04_model_training.ipynb)
- ✅ All model artifacts saved to `src/models/artifacts/`

---

## Training Environment Setup

```bash
# If using Google Colab for LSTM (recommended):
# 1. Upload data/processed/train.csv, val.csv, test.csv to Colab
# 2. Install: pip install xgboost shap torch torchvision
# 3. Enable GPU: Runtime → Change runtime type → T4 GPU

# Local training (XGBoost only):
# XGBoost CPU training for 800 patients × 89 features ~ 20 minutes
# LSTM CPU training: ~3 hours (use Colab instead)
```

---

## 🤖 Vibe Prompt 1 — XGBoost Model

```
Create src/models/xgboost_model.py with XGBoostAEPredictor class.
(AE = Adverse Event)

The model predicts P(adverse glucose event in next 120 minutes) 
using 89 engineered features from EHR + CGM + wearables.

class XGBoostAEPredictor:
    
    DEFAULT_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": ["aucpr", "logloss"],
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 5.7,  # 85/15 class imbalance
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",    # Fast histogram method
        "device": "cpu",
    }

    def __init__(self, params: dict = None):
        self.params = params or self.DEFAULT_PARAMS
        self.model = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series) → dict:
        """
        Train with early stopping on validation AUC-PR.
        Returns training metrics dict.
        """
        self.feature_names = list(X_train.columns)
        
        dtrain = xgb.DMatrix(X_train, label=y_train, 
                              feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val,
                           feature_names=self.feature_names)
        
        evals_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=10,
        )
        self.is_fitted = True
        return evals_result

    def predict_proba(self, X: pd.DataFrame) → np.ndarray:
        """Return P(adverse event) for each row."""
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dmatrix)

    def save(self, path: str) → None:
        """Save model to pickle + XGBoost native format."""
        self.model.save_model(path + ".json")
        import pickle
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) → "XGBoostAEPredictor":
        import pickle
        with open(path + ".pkl", "rb") as f:
            return pickle.load(f)

    def get_feature_importance(self, top_n: int = 20) → pd.DataFrame:
        """Return top N features by gain importance."""
        scores = self.model.get_score(importance_type="gain")
        df = pd.DataFrame({"feature": list(scores.keys()),
                           "importance": list(scores.values())})
        return df.sort_values("importance", ascending=False).head(top_n)

Also create a training script: scripts/train_xgboost.py that:
1. Loads train.csv, val.csv, test.csv
2. Separates features from label column "adverse_event"
3. Trains the model
4. Evaluates on test set: AUC-ROC, AUC-PR, Precision, Recall, F1
5. Saves model to src/models/artifacts/xgboost_v1.pkl
6. Prints feature importance top 20
7. Asserts AUC-ROC >= 0.85 (fail fast if below target)
```

---

## 🤖 Vibe Prompt 2 — LSTM Temporal Model

```
Create src/models/lstm_model.py with LSTMAEPredictor (PyTorch).

The model takes 4-hour CGM + HRV + activity sequences to predict 
P(adverse event in next 120 minutes).

Input shape: (batch_size, 48, 3)
  - 48 = 4 hours × 12 readings/hour (5-min intervals)
  - 3 channels: [glucose_mgdl_normalized, hrv_rmssd_normalized, activity_encoded]

import torch
import torch.nn as nn

class LSTMAEPredictorNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,  # Causal (no future data leakage)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # Use only last timestep output
        out = self.dropout(last_step)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out.squeeze(-1)

class LSTMAEPredictor:
    """Wrapper around LSTMAEPredictorNet for training, inference, save/load."""

    def __init__(self, device: str = "auto"):
        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) 
            else "cpu"
        )
        self.net = LSTMAEPredictorNet().to(self.device)
        self.scaler_glucose = None  # StandardScaler
        self.scaler_hrv = None
        self.is_fitted = False

    def _prepare_sequences(self, df: pd.DataFrame) → torch.Tensor:
        """
        Convert DataFrame to (N, 48, 3) tensor.
        Groups by patient_id + window_id to create sequences.
        Normalizes each channel with its scaler.
        """

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            epochs: int = 50, batch_size: int = 256,
            learning_rate: float = 1e-3) → dict:
        """
        Train with:
        - Binary cross-entropy with pos_weight for class imbalance
        - Adam optimizer with cosine annealing LR schedule
        - Early stopping (patience=10)
        - Returns training history dict
        """
        # pos_weight = negative_count / positive_count
        # criterion = nn.BCEWithLogitsLoss(pos_weight=...)

    def predict_proba(self, X: pd.DataFrame) → np.ndarray:
        """Return P(adverse event) for each sequence."""

    def save(self, path: str) → None:
        torch.save({
            "model_state_dict": self.net.state_dict(),
            "scaler_glucose": self.scaler_glucose,
            "scaler_hrv": self.scaler_hrv,
        }, path)

    @classmethod
    def load(cls, path: str) → "LSTMAEPredictor":
        checkpoint = torch.load(path, map_location="cpu")
        predictor = cls(device="cpu")
        predictor.net.load_state_dict(checkpoint["model_state_dict"])
        predictor.scaler_glucose = checkpoint["scaler_glucose"]
        predictor.scaler_hrv = checkpoint["scaler_hrv"]
        predictor.is_fitted = True
        return predictor

Create scripts/train_lstm.py analogous to train_xgboost.py.
Include a plot_training_curves() function that saves loss + AUC curves.
```

---

## 🤖 Vibe Prompt 3 — Ensemble Model

```
Create src/models/ensemble.py with EnsemblePredictor.

class EnsemblePredictor:
    """
    Weighted ensemble of XGBoost (tabular) + LSTM (temporal) models.
    Uses adaptive weights based on patient's insulin dependency.
    """

    def __init__(self, xgb_model: XGBoostAEPredictor,
                 lstm_model: LSTMAEPredictor):
        self.xgb = xgb_model
        self.lstm = lstm_model

    def predict(self, ehr_features: pd.DataFrame,
                sequence: pd.DataFrame,
                patient_profile: dict) → float:
        """
        Returns final ensemble risk score (0-1).
        
        Adaptive weighting:
          Insulin-dependent patients: XGBoost 35%, LSTM 65%
            (temporal insulin patterns more important)
          Non-insulin patients: XGBoost 60%, LSTM 40%
            (EHR comorbidities more predictive)
        """
        xgb_prob = self.xgb.predict_proba(ehr_features)[0]
        lstm_prob = self.lstm.predict_proba(sequence)[0]

        if patient_profile.get("on_insulin", False):
            weights = (0.35, 0.65)
        else:
            weights = (0.60, 0.40)

        ensemble_score = weights[0] * xgb_prob + weights[1] * lstm_prob
        return float(ensemble_score)

    def predict_with_confidence(self, ehr_features, sequence,
                                 patient_profile) → tuple[float, float, float]:
        """Returns (mean_score, lower_ci, upper_ci) via Monte Carlo dropout."""
        # Run LSTM with dropout enabled N=20 times (MC Dropout)
        # Return mean ± 1.96 × std as confidence interval

    def calibrate(self, X_val_ehr, X_val_seq, y_val, 
                  patient_profiles) → None:
        """
        Platt scaling calibration to ensure probabilities are well-calibrated.
        Critical for clinical use — a "0.8 risk" should mean 80% of patients
        with that score actually have an event.
        """
        from sklearn.calibration import CalibratedClassifierCV
        # Fit isotonic regression calibrator on validation set

    def evaluate(self, X_test_ehr, X_test_seq, y_test,
                  patient_profiles) → dict:
        """Full evaluation metrics on test set."""
        probs = [self.predict(xehr, xseq, prof) 
                 for xehr, xseq, prof in zip(...)]
        preds = [1 if p >= 0.5 else 0 for p in probs]
        
        from sklearn.metrics import (roc_auc_score, average_precision_score,
                                     precision_score, recall_score, f1_score,
                                     confusion_matrix)
        return {
            "auc_roc": roc_auc_score(y_test, probs),
            "auc_pr": average_precision_score(y_test, probs),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }

Implement all methods fully with type hints and docstrings.
```

---

## 🤖 Vibe Prompt 4 — SHAP Explainability

```
Create src/explainability/shap_explainer.py with SHAPExplainer.

class SHAPExplainer:
    """
    Provides SHAP-based explanations for XGBoost predictions.
    Used to generate clinically interpretable alerts.
    """

    def __init__(self, xgb_model: XGBoostAEPredictor):
        import shap
        self.explainer = shap.TreeExplainer(xgb_model.model)
        self.feature_names = xgb_model.feature_names

    def explain_prediction(self, X: pd.DataFrame,
                            top_n: int = 10) → List[dict]:
        """
        Compute SHAP values for a single prediction.
        Returns top N features sorted by |SHAP value|.
        
        Output format:
        [
            {
                "feature": "glucose_roc_current",
                "value": 3.2,                 # Actual feature value
                "shap_value": 0.18,           # Contribution to prediction
                "direction": "increases_risk", # pos/neg
                "clinical_label": "Glucose rising rapidly (+3.2 mg/dL/min)",
            },
            ...
        ]
        """
        shap_values = self.explainer.shap_values(X)
        feature_impacts = []
        for i, fname in enumerate(self.feature_names):
            feature_impacts.append({
                "feature": fname,
                "value": float(X.iloc[0][fname]),
                "shap_value": float(shap_values[0][i]),
                "direction": "increases_risk" if shap_values[0][i] > 0 else "decreases_risk",
                "clinical_label": self._get_clinical_label(fname, X.iloc[0][fname]),
            })
        
        # Sort by absolute SHAP value
        feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return feature_impacts[:top_n]

    def _get_clinical_label(self, feature_name: str, value: float) → str:
        """Convert feature name + value to human-readable clinical label."""
        label_map = {
            "glucose_roc_current": f"Glucose rate of change: {value:+.1f} mg/dL/min",
            "hrv_rmssd_ms": f"HRV (RMSSD): {value:.0f} ms",
            "hrv_suppression_flag": "HRV suppression detected" if value > 0.5 else "Normal HRV",
            "glucose_current": f"Current glucose: {value:.0f} mg/dL",
            "glucose_std_4h": f"Glucose variability (4h): ±{value:.1f} mg/dL",
            "io_board": f"Insulin on board: {value:.1f} units",
            "time_since_last_meal_min": f"Time since last meal: {value:.0f} min",
            "is_exercising": "Currently exercising" if value > 0.5 else "Not exercising",
            "tir_24h": f"Time in range (24h): {value*100:.0f}%",
            "hba1c_latest": f"HbA1c: {value:.1f}%",
        }
        return label_map.get(feature_name, f"{feature_name}: {value:.2f}")

    def generate_waterfall_data(self, X: pd.DataFrame) → dict:
        """
        Generate data for a SHAP waterfall chart in the React dashboard.
        Returns JSON-serializable dict for the frontend.
        """
        features = self.explain_prediction(X, top_n=10)
        return {
            "base_value": float(self.explainer.expected_value),
            "final_value": float(self.explainer.shap_values(X).sum() + 
                                  self.explainer.expected_value),
            "features": features,
        }

Also create scripts/fit_shap.py that:
1. Loads the trained XGBoost model
2. Creates a background SHAP dataset (100 random training samples)
3. Fits the TreeExplainer
4. Saves the explainer to src/models/artifacts/shap_explainer.pkl
5. Runs a test explanation and prints top 5 features
```

---

## 🤖 Vibe Prompt 5 — Model Evaluation Notebook

```
Create notebooks/04_model_training.ipynb (and 05_evaluation.ipynb).

notebook 04 — Training Walkthrough:
1. Load train/val/test datasets
2. Class balance visualization (pie + bar)
3. Feature correlation heatmap (89×89 — use hierarchical clustering)
4. Train XGBoost — plot training AUC-PR curve
5. Train LSTM — plot training + validation loss curves
6. Train Ensemble — combine models

notebook 05 — Evaluation Dashboard:
1. ROC Curve comparison: XGBoost vs LSTM vs Ensemble (3 curves on one plot)
2. Precision-Recall Curve (more informative for imbalanced data)
3. Confusion Matrix (absolute counts + normalized %)
4. Calibration Plot (reliability diagram): 
   "When model says 80% risk, does 80% of patients actually have an event?"
5. Performance by Subgroup:
   - By HbA1c category (<7, 7-9, >9)
   - By Age group (35-45, 45-55, 55-65, 65+)
   - By Insulin status (yes/no)
   - By Gender (bias check)
6. SHAP Summary Plot (beeswarm — shows feature importance across all test patients)
7. SHAP Dependence Plots for top 5 features
8. Example predictions: 3 TRUE POSITIVES with SHAP waterfall
9. Error Analysis: False Positives — what patterns cause them?
10. Final metrics table (for README + presentation)

Include markdown cells with clinical interpretation of each finding.
Save final metrics to data/processed/model_metrics.json
```

---

## ✅ Training Pipeline Execution Order

```bash
# Step 1: Build dataset (from Session 02)
python scripts/build_dataset.py

# Step 2: Train XGBoost (local, ~20 min)
python scripts/train_xgboost.py
# Expected output:
# [0] train-aucpr:0.823  val-aucpr:0.801
# [50] train-aucpr:0.893  val-aucpr:0.871
# Best round: 312
# Test AUC-ROC: 0.891 ✅

# Step 3: Train LSTM (Colab or local GPU, ~45 min on Colab T4)
python scripts/train_lstm.py
# Expected: Test AUC-ROC: 0.856

# Step 4: Evaluate ensemble
python scripts/evaluate_ensemble.py
# Expected: Ensemble AUC-ROC: 0.907 ✅

# Step 5: Fit SHAP
python scripts/fit_shap.py

# Verify artifacts exist
ls src/models/artifacts/
# xgboost_v1.json  xgboost_v1.pkl  lstm_v1.pt  shap_explainer.pkl
```

---

## 📊 Expected Performance Table

| Model | AUC-ROC | AUC-PR | Precision | Recall | F1 |
|---|---|---|---|---|---|
| XGBoost | 0.891 | 0.742 | 0.84 | 0.79 | 0.815 |
| LSTM | 0.856 | 0.701 | 0.81 | 0.77 | 0.789 |
| **Ensemble** | **0.907** | **0.779** | **0.87** | **0.83** | **0.849** |

---

## 📋 Session 06 Completion Checklist

- [ ] `xgboost_model.py` complete with train/predict/save/load
- [ ] `lstm_model.py` complete with PyTorch net + training loop
- [ ] `ensemble.py` with adaptive weighting + calibration
- [ ] `shap_explainer.py` with clinical labels
- [ ] All 4 model artifacts saved in `src/models/artifacts/`
- [ ] Ensemble AUC-ROC ≥ 0.88 on test set
- [ ] Evaluation notebook shows calibration + subgroup analysis
- [ ] SHAP waterfall example works for a sample patient

## 🔜 Next Session

→ **07-prediction-engine.md** — Wiring models into the PatientTwin prediction loop

---

*GlucoTwin Vibe Assist 06 | ML Model Training | Digital Twin Challenge 2026*
