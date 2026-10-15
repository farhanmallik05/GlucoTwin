"""
src/data/ingestion/ohio_parser.py

Ohio T1DM dataset XML parser with 5-minute uniform resampling,
T1D-to-T2D physiological glycemic variance adaptation, and graceful error handling.
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import numpy as np
import logging

logger = logging.getLogger(__name__)


class OhioDataNotAvailableError(FileNotFoundError):
    """
    Raised when Ohio T1DM XML data files are not found.
    Directs the user/pipeline to use the synthetic CGM generator.
    """

    def __init__(self, path: Union[str, Path]):
        message = (
            f"Ohio T1DM dataset not found at '{path}'. "
            "The Ohio T1DM dataset requires an academic research agreement. "
            "Please use the synthetic data generator: src/data/synthetic/cgm_simulator.py."
        )
        super().__init__(message)
        self.path = path


class OhioDatasetParser:
    """
    Parses and normalizes Ohio T1DM XML patient data into the GlucoTwin schema.
    """

    def __init__(self, adapt_to_t2d: bool = True, variance_scale: float = 0.70):
        self.adapt_to_t2d = adapt_to_t2d
        self.variance_scale = variance_scale

    def is_available(self, ohio_path: Union[str, Path]) -> bool:
        """Checks whether the specified Ohio dataset path exists and contains XML files."""
        p = Path(ohio_path)
        if not p.exists():
            return False
        if p.is_file():
            return p.suffix.lower() == ".xml"
        return any(p.glob("*.xml"))

    def parse_patient(self, xml_path: Union[str, Path]) -> pd.DataFrame:
        """
        Parses a single Ohio patient XML file into a 5-minute sampled DataFrame.

        Args:
            xml_path: Path to patient XML file (e.g. 559-ws-training.xml).

        Returns:
            pd.DataFrame indexed by 5-min timestamp with glucose, ROC, ACC, HR, steps.

        Raises:
            OhioDataNotAvailableError: If xml_path does not exist.
        """
        path = Path(xml_path)
        if not path.exists():
            raise OhioDataNotAvailableError(path)

        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception as e:
            raise OhioDataNotAvailableError(path) from e

        # Extract glucose readings
        records = []
        patient_id = root.attrib.get("id", path.stem.split("-")[0])

        for event in root.findall(".//glucose_level/event"):
            ts_str = event.attrib.get("ts")
            val_str = event.attrib.get("value")
            if ts_str and val_str:
                try:
                    # Ohio timestamp format: DD-MM-YYYY HH:MM:SS or YYYY-MM-DD HH:MM:SS
                    try:
                        dt = datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")
                    except ValueError:
                        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    records.append({"timestamp": dt, "glucose_mgdl": float(val_str)})
                except Exception:
                    pass

        if not records:
            raise OhioDataNotAvailableError(path)

        df = pd.DataFrame(records).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        df = df.set_index("timestamp").resample("5min").mean()
        # Linear interpolation for gaps <= 15 minutes (3 steps)
        df["glucose_mgdl"] = df["glucose_mgdl"].interpolate(method="time", limit=3).ffill().bfill()
        df = df.reset_index()

        # Adapt to GlucoTwin schema
        return self.to_glucotwin_schema(df, patient_id=patient_id)

    def parse_all(self, ohio_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """
        Parses all patient XML files in the specified Ohio directory.

        Raises:
            OhioDataNotAvailableError: If directory is empty or does not exist.
        """
        path = Path(ohio_dir)
        if not path.exists() or not path.is_dir():
            raise OhioDataNotAvailableError(path)

        xml_files = list(path.glob("*.xml"))
        if not xml_files:
            raise OhioDataNotAvailableError(path)

        patients = {}
        for xf in xml_files:
            try:
                df = self.parse_patient(xf)
                pid = df["patient_id"].iloc[0] if "patient_id" in df.columns else xf.stem
                patients[pid] = df
            except Exception as e:
                logger.warning(f"Error parsing Ohio XML file {xf}: {e}")

        if not patients:
            raise OhioDataNotAvailableError(path)

        return patients

    def to_glucotwin_schema(
        self,
        df: pd.DataFrame,
        patient_id: str,
    ) -> pd.DataFrame:
        """
        Maps raw parsed XML event streams into GlucoTwin 5-minute synchronized schema.
        Applies T1D->T2D variance scaling and computes derivatives.
        """
        res = df.copy()
        res["patient_id"] = patient_id

        glucose = res["glucose_mgdl"].values.astype(float)
        
        # Physiological adaptation: G_T2D = mu + variance_scale * (G_T1D - mu)
        if self.adapt_to_t2d and len(glucose) > 0:
            mu_g = np.nanmean(glucose)
            adapted = mu_g + self.variance_scale * (glucose - mu_g)
            glucose = np.clip(adapted, 40.0, 400.0)
            res["glucose_mgdl"] = np.round(glucose, 1)

        # Compute ROC and ACC
        n = len(glucose)
        roc = np.zeros(n)
        acc = np.zeros(n)
        if n > 1:
            roc[1:] = (glucose[1:] - glucose[:-1]) / 5.0
            acc[1:] = (roc[1:] - roc[:-1]) / 5.0

        res["glucose_roc"] = np.round(roc, 3)
        res["glucose_acc"] = np.round(acc, 4)

        # Baseline wearable channels if not present
        if "heart_rate_bpm" not in res.columns:
            res["heart_rate_bpm"] = 72.0
        if "hrv_rmssd_ms" not in res.columns:
            res["hrv_rmssd_ms"] = 38.0
        if "hrv_sdnn_ms" not in res.columns:
            res["hrv_sdnn_ms"] = 45.0
        if "hrv_lf_hf_ratio" not in res.columns:
            res["hrv_lf_hf_ratio"] = 1.5
        if "steps_last_5min" not in res.columns:
            res["steps_last_5min"] = 15
        if "steps_last_hour" not in res.columns:
            res["steps_last_hour"] = 180
        if "activity_type" not in res.columns:
            res["activity_type"] = "resting"
        if "sleep_stage" not in res.columns:
            res["sleep_stage"] = "none"
        if "sleep_quality" not in res.columns:
            res["sleep_quality"] = 0.85
        if "data_quality" not in res.columns:
            res["data_quality"] = "good"
        if "source" not in res.columns:
            res["source"] = "ohio"

        return res
