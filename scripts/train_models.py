"""
scripts/train_models.py
Trains and serializes baseline GlucoTwin ML models (XGBoost + PyTorch LSTM).
Saves operational model artifacts into src/models/artifacts/.
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.feature_engineering import FEATURE_COLUMNS
from src.models.xgboost_model import XGBoostAEPredictor
from src.models.lstm_model import LSTMAEPredictorNet


def generate_synthetic_training_data(n_samples: int = 500):
    """Generate aligned tabular and sequential training pairs."""
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Tabular features DataFrame (n_samples, 89)
    data = {}
    for col in FEATURE_COLUMNS:
        if "glucose" in col:
            data[col] = np.random.normal(150.0, 30.0, n_samples)
        elif "hrv" in col:
            data[col] = np.random.normal(38.0, 8.0, n_samples)
        elif "hba1c" in col:
            data[col] = np.random.normal(8.2, 1.2, n_samples)
        elif "bmi" in col:
            data[col] = np.random.normal(26.5, 3.5, n_samples)
        elif "age" in col:
            data[col] = np.random.randint(35, 75, n_samples)
        else:
            data[col] = np.random.uniform(0.0, 1.0, n_samples)

    df = pd.DataFrame(data)

    # 2. Ground truth labels (imbalance ~85% normal, 15% adverse events)
    risk_signal = (
        (df["glucose_current"] > 220.0).astype(int) * 3 +
        (df["glucose_roc_current"] > 2.0).astype(int) * 2 +
        (df["hrv_rmssd_mean_30min"] < 25.0).astype(int) * 2 +
        (df["hba1c_latest"] > 8.5).astype(int) * 1.5
    )
    y_prob = 1.0 / (1.0 + np.exp(-(risk_signal - 3.5)))
    y = (y_prob > 0.65).astype(int)

    # Ensure class imbalance roughly 85:15
    print(f"Generated {n_samples} samples with label distribution: {np.bincount(y)}")

    # 3. Sequential tensors for LSTM: shape (n_samples, 48 timesteps, 3 channels)
    # Channels: [normalized glucose, normalized HRV, activity]
    seq_data = np.zeros((n_samples, 48, 3), dtype=np.float32)
    for i in range(n_samples):
        base_g = df["glucose_current"].iloc[i] / 200.0
        base_h = df["hrv_rmssd_mean_30min"].iloc[i] / 50.0
        seq_data[i, :, 0] = base_g + np.cumsum(np.random.normal(0, 0.02, 48))
        seq_data[i, :, 1] = base_h + np.cumsum(np.random.normal(0, 0.01, 48))
        seq_data[i, :, 2] = np.random.choice([0.0, 0.5, 1.0], 48)

    X_seq = torch.tensor(seq_data, dtype=torch.float32)
    y_seq = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    return df, y, X_seq, y_seq


def train_and_save():
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "models", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    print("Step 1: Generating synthetic training data...")
    X_tab, y_tab, X_seq, y_seq = generate_synthetic_training_data(n_samples=600)

    # Train XGBoost
    print("Step 2: Training XGBoost tabular predictor...")
    xgb_predictor = XGBoostAEPredictor()
    xgb_predictor.fit(X_tab, y_tab)

    xgb_pkl_path = os.path.join(artifacts_dir, "xgboost_v1.pkl")
    with open(xgb_pkl_path, "wb") as f:
        pickle.dump(xgb_predictor, f)
    print(f"  -> Saved XGBoost artifact: {xgb_pkl_path}")

    # Train LSTM
    print("Step 3: Training PyTorch causal LSTM network...")
    lstm_net = LSTMAEPredictorNet(input_dim=3, hidden_dim_1=128, hidden_dim_2=64)
    optimizer = torch.optim.Adam(lstm_net.parameters(), lr=1e-3)
    pos_weight = torch.tensor([5.7])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lstm_net.train()
    for epoch in range(15):
        optimizer.zero_grad()
        out = lstm_net(X_seq)
        loss = criterion(out, y_seq)
        loss.backward()
        optimizer.step()

    lstm_net.eval()
    lstm_pt_path = os.path.join(artifacts_dir, "lstm_v1.pt")
    torch.save({"model_state_dict": lstm_net.state_dict()}, lstm_pt_path)
    print(f"  -> Saved PyTorch LSTM artifact: {lstm_pt_path}")

    print("\nModel training and artifact serialization complete!")


if __name__ == "__main__":
    train_and_save()
