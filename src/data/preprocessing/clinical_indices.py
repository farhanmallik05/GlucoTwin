"""
src/data/preprocessing/clinical_indices.py

Consensus glycemic indices calculation engine conforming to international
ADA/EASD/ATTD clinical guidelines:
- TIR (Time In Range, 70-180 mg/dL)
- TAR (Time Above Range, >180 mg/dL)
- TBR (Time Below Range, <70 mg/dL)
- GMI (Glucose Management Indicator %, Berger et al. 2018)
- CV (Glucose Coefficient of Variation %, SD / Mean * 100)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def _clean_glucose_array(glucose: Union[pd.Series, np.ndarray, List[float]]) -> np.ndarray:
    """Extracts valid finite numeric readings from Series, array, or list."""
    if isinstance(glucose, pd.Series):
        arr = glucose.to_numpy(dtype=float, copy=True)
    elif isinstance(glucose, np.ndarray):
        arr = glucose.astype(float, copy=True)
    elif isinstance(glucose, (list, tuple)):
        arr = np.array(glucose, dtype=float)
    else:
        arr = np.array([glucose], dtype=float)

    # Filter out NaNs, Infinities, and non-positive anomalies
    valid_mask = np.isfinite(arr) & (arr > 0)
    return arr[valid_mask]


def calculate_tir(
    glucose: Union[pd.Series, np.ndarray, List[float]],
    low: float = 70.0,
    high: float = 180.0,
    as_percentage: bool = False,
) -> float:
    """
    Calculates fraction/percentage of glucose readings between low and high (inclusive).
    
    Args:
        glucose: Sequence of glucose readings in mg/dL.
        low: Lower bound threshold in mg/dL (default 70.0).
        high: Upper bound threshold in mg/dL (default 180.0).
        as_percentage: If True, returns value in [0.0, 100.0]; else in [0.0, 1.0].
        
    Returns:
        TIR as float. Returns 1.0 (or 100.0) if input array is empty.
    """
    clean = _clean_glucose_array(glucose)
    if len(clean) == 0:
        return 100.0 if as_percentage else 1.0
    val = float(np.mean((clean >= low) & (clean <= high)))
    return val * 100.0 if as_percentage else val


def calculate_tar(
    glucose: Union[pd.Series, np.ndarray, List[float]],
    threshold: float = 180.0,
    as_percentage: bool = False,
) -> float:
    """
    Calculates fraction/percentage of glucose readings strictly above threshold.
    
    Args:
        glucose: Sequence of glucose readings in mg/dL.
        threshold: Hyperglycemic threshold in mg/dL (default 180.0).
        as_percentage: If True, returns value in [0.0, 100.0]; else in [0.0, 1.0].
        
    Returns:
        TAR as float. Returns 0.0 if input array is empty.
    """
    clean = _clean_glucose_array(glucose)
    if len(clean) == 0:
        return 0.0
    val = float(np.mean(clean > threshold))
    return val * 100.0 if as_percentage else val


def calculate_tbr(
    glucose: Union[pd.Series, np.ndarray, List[float]],
    threshold: float = 70.0,
    as_percentage: bool = False,
) -> float:
    """
    Calculates fraction/percentage of glucose readings strictly below threshold.
    
    Args:
        glucose: Sequence of glucose readings in mg/dL.
        threshold: Hypoglycemic threshold in mg/dL (default 70.0).
        as_percentage: If True, returns value in [0.0, 100.0]; else in [0.0, 1.0].
        
    Returns:
        TBR as float. Returns 0.0 if input array is empty.
    """
    clean = _clean_glucose_array(glucose)
    if len(clean) == 0:
        return 0.0
    val = float(np.mean(clean < threshold))
    return val * 100.0 if as_percentage else val


def calculate_gmi(glucose: Union[pd.Series, np.ndarray, List[float]]) -> float:
    """
    Calculates Glucose Management Indicator (%) using standard 2018 Berger formula:
    GMI (%) = 3.31 + 0.02392 * mean_glucose (mg/dL)
    
    Returns:
        Estimated HbA1c equivalent %. Default 5.7% (normal baseline) if empty.
    """
    clean = _clean_glucose_array(glucose)
    if len(clean) == 0:
        return 5.7
    mean_glucose = float(np.mean(clean))
    return float(3.31 + 0.02392 * mean_glucose)


def calculate_cv(glucose: Union[pd.Series, np.ndarray, List[float]]) -> float:
    """
    Calculates Glucose Coefficient of Variation (%) = (std / mean) * 100.
    
    Returns:
        CV % as float. Returns 0.0 if len < 2 or mean <= 0.
    """
    clean = _clean_glucose_array(glucose)
    if len(clean) < 2:
        return 0.0
    mean_g = float(np.mean(clean))
    if mean_g <= 1e-6:
        return 0.0
    std_g = float(np.std(clean, ddof=1))
    return float((std_g / mean_g) * 100.0)


def compute_clinical_indices(
    glucose: Union[pd.Series, np.ndarray, List[float]],
    as_percentage: bool = True,
) -> Dict[str, float]:
    """
    Computes all 5 standard clinical indices in a single dictionary.
    
    Args:
        glucose: Sequence of glucose readings.
        as_percentage: If True, TIR, TAR, and TBR are returned in [0.0, 100.0].
        
    Returns:
        Dictionary with keys 'tir_24h', 'tar_24h', 'tbr_24h', 'gmi', 'cv_glucose_24h'.
    """
    return {
        "tir_24h": calculate_tir(glucose, as_percentage=as_percentage),
        "tar_24h": calculate_tar(glucose, as_percentage=as_percentage),
        "tbr_24h": calculate_tbr(glucose, as_percentage=as_percentage),
        "gmi": calculate_gmi(glucose),
        "cv_glucose_24h": calculate_cv(glucose),
    }


class ClinicalIndicesCalculator:
    """
    Class-based interface for computing clinical glycemic indices,
    matching test suite and digital twin expectations.
    """

    def __init__(self, as_percentage: bool = True):
        self.as_percentage = as_percentage

    def compute(
        self,
        glucose: Union[pd.Series, np.ndarray, List[float]],
        as_percentage: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Calculates all clinical indices for the given glucose series.
        """
        use_pct = self.as_percentage if as_percentage is None else as_percentage
        return compute_clinical_indices(glucose, as_percentage=use_pct)
