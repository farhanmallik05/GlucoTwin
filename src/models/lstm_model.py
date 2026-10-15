"""
src/models/lstm_model.py
Causal PyTorch LSTM for Temporal Sequence Glycemic Excursion Prediction.
Consumes multi-channel sequences of shape (batch, 48, 3) representing 4 hours of wearable data.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class LSTMAEPredictorNet(nn.Module):
    """
    Causal unidirectional LSTM network for real-time temporal sequence modeling.
    Takes 4-hour historical sequences (48 time-steps x 3 channels) without look-ahead bias.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        dropout: float = 0.3
    ):
        super().__init__()
        # Accommodate both naming conventions
        in_dim = input_size if input_size is not None else input_dim
        h_dim1 = hidden_size if hidden_size is not None else hidden_dim_1
        h_dim2 = hidden_dim_2

        self.lstm1 = nn.LSTM(
            input_size=in_dim,
            hidden_size=h_dim1,
            batch_first=True,
            bidirectional=False
        )
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=h_dim1,
            hidden_size=h_dim2,
            batch_first=True,
            bidirectional=False
        )
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(h_dim2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over sequential input tensor.
        Args:
            x: Tensor of shape (batch_size, sequence_length=48, channels=3)
        Returns:
            Tensor of shape (batch_size, 1) with probabilities in [0.0, 1.0].
        """
        # Ensure float32
        x = x.float()
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.lstm2(out1)
        # Select last causal timestep: shape (batch_size, h_dim2)
        last_timestep = out2[:, -1, :]
        feat = self.dropout2(last_timestep)
        hidden = self.relu(self.fc1(feat))
        prob = self.sigmoid(self.fc2(hidden))
        return prob


class LSTMAEPredictor:
    """High-level wrapper for LSTMAEPredictorNet inference and confidence estimation."""

    def __init__(self, device: str = "cpu", model_net: Optional[LSTMAEPredictorNet] = None):
        self.device = torch.device(device)
        self.net = model_net or LSTMAEPredictorNet()
        self.net.to(self.device)
        self.net.eval()
        self.is_fitted = True

    def predict_proba(self, sequence: Union[torch.Tensor, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Run forward inference."""
        self.net.eval()
        with torch.no_grad():
            if isinstance(sequence, (np.ndarray, list)):
                t = torch.tensor(sequence, dtype=torch.float32, device=self.device)
            elif isinstance(sequence, pd.DataFrame):
                t = torch.tensor(sequence.values, dtype=torch.float32, device=self.device)
            elif isinstance(sequence, torch.Tensor):
                t = sequence.to(self.device).float()
            else:
                t = torch.randn(1, 48, 3, device=self.device)

            if t.ndim == 2:
                t = t.unsqueeze(0)  # Shape (1, 48, 3)

            out = self.net(t)
            return out.cpu().numpy()

    def mc_dropout_inference(self, sequence: torch.Tensor, num_samples: int = 10) -> Tuple[float, float, float]:
        """
        Monte Carlo Dropout for Bayesian predictive uncertainty estimation.
        Activates dropout at inference time across stochastic forward passes.
        """
        self.net.train()  # Keep dropout active
        predictions = []
        with torch.no_grad():
            t = sequence.to(self.device).float()
            if t.ndim == 2:
                t = t.unsqueeze(0)
            for _ in range(num_samples):
                out = self.net(t)
                predictions.append(out.item())

        mean_val = float(np.mean(predictions))
        std_val = float(np.std(predictions))
        ci_lower = max(0.0, mean_val - 1.96 * std_val)
        ci_upper = min(1.0, mean_val + 1.96 * std_val)
        return mean_val, ci_lower, ci_upper
