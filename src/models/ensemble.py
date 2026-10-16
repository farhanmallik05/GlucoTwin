"""
src/models/ensemble.py
Patient-Adaptive Weighted Ensemble of XGBoost (Tabular) and PyTorch LSTM (Temporal).
Dynamically adjusts weights based on patient pharmacological context (Insulin vs Oral).
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from src.models.xgboost_model import XGBoostAEPredictor
from src.models.lstm_model import LSTMAEPredictor, LSTMAEPredictorNet


class EnsemblePredictor:
    """
    Multimodal fusion predictor combining tabular baseline risk with dynamic time-series trends.
    Uses clinical context-aware adaptive weighting:
    - Insulin patients: 35% Tabular + 65% Temporal (insulin pharmacodynamics dominate)
    - Non-insulin patients: 60% Tabular + 40% Temporal (EHR comorbidities & organ function dominate)
    """

    def __init__(
        self,
        xgb_model: Optional[XGBoostAEPredictor] = None,
        lstm_model: Optional[LSTMAEPredictor] = None
    ):
        self.xgb = xgb_model or XGBoostAEPredictor()
        self.lstm = lstm_model or LSTMAEPredictor()

    def get_weights(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Derive adaptive model weights based on patient insulin dependency status.
        """
        ctx = context or {}
        on_insulin = bool(ctx.get("on_insulin", False))
        if on_insulin:
            return {"xgboost": 0.35, "lstm": 0.65}
        return {"xgboost": 0.60, "lstm": 0.40}

    def predict(self, *args, **kwargs) -> float:
        """
        Execute prediction. Supports direct probability combination:
            predict(p_xgb: float, p_lstm: float, context: dict) -> float
        as well as raw feature ingestion:
            predict(ehr_df, seq_tensor, context: dict) -> float
        """
        if len(args) >= 3 and isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)):
            p_xgb = float(args[0])
            p_lstm = float(args[1])
            context = args[2] if isinstance(args[2], dict) else {}
        elif len(args) >= 2 and isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)):
            p_xgb = float(args[0])
            p_lstm = float(args[1])
            context = kwargs.get("context", {})
        else:
            # Multi-modal input branch
            ehr_input = args[0] if len(args) > 0 else kwargs.get("ehr")
            seq_input = args[1] if len(args) > 1 else kwargs.get("seq")
            context = args[2] if len(args) > 2 else kwargs.get("context", {})

            p_xgb = float(self.xgb.predict_proba(ehr_input))
            p_lstm_raw = self.lstm.predict_proba(seq_input)
            p_lstm = float(p_lstm_raw[0, 0] if p_lstm_raw.ndim == 2 else p_lstm_raw[0])

        weights = self.get_weights(context)
        final_score = (weights["xgboost"] * p_xgb) + (weights["lstm"] * p_lstm)
        return float(np.clip(final_score, 0.0, 1.0))

    def estimate_confidence_interval(
        self,
        sequence: Union[torch.Tensor, np.ndarray],
        num_samples: int = 30
    ) -> Tuple[float, float]:
        """
        Estimate 95% predictive confidence intervals via Monte Carlo Dropout.
        """
        if isinstance(sequence, np.ndarray):
            seq_tensor = torch.tensor(sequence, dtype=torch.float32)
        else:
            seq_tensor = sequence

        mean_val, ci_lower, ci_upper = self.lstm.mc_dropout_inference(seq_tensor, num_samples=num_samples)
        # Ensure ordering and bounds
        ci_lower = float(np.clip(ci_lower, 0.0, 1.0))
        ci_upper = float(np.clip(ci_upper, 0.0, 1.0))
        if ci_lower > ci_upper:
            ci_lower, ci_upper = ci_upper, ci_lower
        return ci_lower, ci_upper
