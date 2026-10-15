"""
scripts/build_dataset.py

End-to-end dataset generation pipeline:
1. Generates or loads Indian T2D EHR cohort profiles (23 features).
2. Generates 5-minute CGM trajectories via CGMSimulator + WearableSimulator.
3. Labels adverse events across 120-minute forward horizon (24 steps).
4. Extracts 89 engineered features across 4-hour sliding windows (48 steps).
5. Partitions dataset into train (80%), validation (10%), and test (10%) splits
   strictly by patient ID (zero leakage).
6. Exports train.csv, val.csv, test.csv, and dataset_stats.json.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic.patient_generator import PatientGenerator, EdgeCaseGenerator
from src.data.synthetic.cgm_simulator import CGMSimulator
from src.data.ingestion.wearable_simulator import WearableSimulator
from src.data.preprocessing.labeler import label_adverse_events
from src.data.preprocessing.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.data.preprocessing.clinical_validator import ClinicalValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_dataset")


def process_single_patient(
    patient_ehr: Dict[str, Any],
    duration_days: int,
    seed: int,
) -> pd.DataFrame:
    """
    Worker function to process one patient end-to-end:
    CGM sim -> Wearable sim -> Event labeling -> 89-feature window extraction.
    """
    pid = patient_ehr["patient_id"]

    # 1. Generate 5-min CGM time series
    cgm_sim = CGMSimulator(patient_ehr=patient_ehr, seed=seed)
    cgm_df = cgm_sim.generate(patient_ehr=patient_ehr, duration_days=duration_days, add_noise=True)

    # 2. Add wearable streams (HRV, steps, sleep)
    wearable_sim = WearableSimulator(patient_ehr=patient_ehr, seed=seed)
    combined_df = wearable_sim.generate(patient_ehr=patient_ehr, cgm_df=cgm_df)

    # 3. Label adverse events (120-min forward horizon = 24 steps)
    labeled_df = label_adverse_events(combined_df, horizon_minutes=120)

    # 4. Extract 89 features for rolling windows of length 48 (4 hours)
    feat_engineer = FeatureEngineer()
    n_rows = len(labeled_df)
    window_size = 48  # 4 hours

    records = []
    # Drop rows without full 120-min forward lookahead (last 24 steps)
    valid_end = n_rows - 24

    for t in range(window_size, valid_end):
        window_slice = labeled_df.iloc[t - window_size : t]
        feat_dict = feat_engineer.compute_features(window_slice, patient_ehr)

        # Attach ground truth and metadata
        feat_dict["patient_id"] = pid
        feat_dict["timestamp"] = labeled_df.iloc[t]["timestamp"]
        feat_dict["adverse_event"] = int(labeled_df.iloc[t]["adverse_event"])
        records.append(feat_dict)

    if not records:
        return pd.DataFrame()

    patient_dataset_df = pd.DataFrame(records)
    return patient_dataset_df


def main():
    parser = argparse.ArgumentParser(description="GlucoTwin Dataset Builder")
    parser.add_argument("--num-patients", type=int, default=100, help="Number of synthetic patients")
    parser.add_argument("--days", type=int, default=14, help="Simulation duration in days per patient")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (-1 for all cores)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (10 patients, 3 days) for fast testing")
    args = parser.parse_args()

    if args.quick:
        logger.info("⚡ Quick mode activated: using 10 patients, 3 days")
        num_patients = 10
        duration_days = 3
    else:
        num_patients = args.num_patients
        duration_days = args.days

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate synthetic population
    logger.info(f"Generating {num_patients} Indian T2D patient EHR profiles...")
    pat_gen = PatientGenerator(seed=args.seed)
    ehr_df = pat_gen.generate_population(n_patients=num_patients, include_edge_cases=True)

    # Step 2: Validate EHR profiles
    validator = ClinicalValidator()
    val_report = validator.validate_dataset(ehr_df)
    logger.info(f"EHR Clinical Validation: {val_report.valid_percentage:.1f}% valid, {val_report.error_count} errors")
    if val_report.error_count > 0:
        logger.warning("Filtering out invalid EHR profiles...")
        ehr_df = val_report.cleaned_df

    patient_records = ehr_df.to_dict(orient="records")

    # Step 3: Parallel processing across patients
    logger.info(f"Simulating streams and computing 89 features for {len(patient_records)} patients across {duration_days} days...")
    # Use 1 worker if n-jobs is 1 or in quick mode on constrained systems, otherwise args.n_jobs
    n_jobs = 1 if num_patients <= 10 else args.n_jobs
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_patient)(p, duration_days, args.seed + idx)
        for idx, p in enumerate(tqdm(patient_records, desc="Processing Patients"))
    )

    valid_results = [r for r in results if not r.empty]
    if not valid_results:
        raise RuntimeError("No valid patient records could be processed.")

    full_dataset = pd.concat(valid_results, ignore_index=True)
    logger.info(f"Total processed samples: {len(full_dataset):,}")

    # Step 4: Stratified Patient-Level Splitting (80/10/10)
    unique_pids = list(full_dataset["patient_id"].unique())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_pids)

    n_total = len(unique_pids)
    n_train = max(1, int(0.80 * n_total))
    n_val = max(1, int(0.10 * n_total)) if n_total > 2 else 0

    train_pids = set(unique_pids[:n_train])
    val_pids = set(unique_pids[n_train : n_train + n_val])
    test_pids = set(unique_pids[n_train + n_val :])

    # If small number of patients, ensure test has at least 1 if possible
    if len(test_pids) == 0 and len(train_pids) > 1:
        moved = train_pids.pop()
        test_pids.add(moved)

    # STRICT DATA LEAKAGE ASSERTIONS
    assert len(train_pids & val_pids) == 0, "Leakage detected between train and val!"
    assert len(train_pids & test_pids) == 0, "Leakage detected between train and test!"
    assert len(val_pids & test_pids) == 0, "Leakage detected between val and test!"

    train_df = full_dataset[full_dataset["patient_id"].isin(train_pids)].copy()
    val_df = full_dataset[full_dataset["patient_id"].isin(val_pids)].copy() if val_pids else pd.DataFrame(columns=full_dataset.columns)
    test_df = full_dataset[full_dataset["patient_id"].isin(test_pids)].copy() if test_pids else pd.DataFrame(columns=full_dataset.columns)

    # Step 5: Ensure exact 89 feature columns + target
    cols_to_save = ["patient_id", "timestamp"] + FEATURE_COLUMNS + ["adverse_event"]
    train_df = train_df[cols_to_save]
    if not val_df.empty:
        val_df = val_df[cols_to_save]
    if not test_df.empty:
        test_df = test_df[cols_to_save]

    # Step 6: Export CSVs
    logger.info("Saving datasets to disk...")
    train_file = output_path / "train.csv"
    val_file = output_path / "val.csv"
    test_file = output_path / "test.csv"

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    # Step 7: Export dataset stats
    overall_ae_rate = float(full_dataset["adverse_event"].mean())
    train_ae_rate = float(train_df["adverse_event"].mean()) if not train_df.empty else 0.0
    val_ae_rate = float(val_df["adverse_event"].mean()) if not val_df.empty else 0.0
    test_ae_rate = float(test_df["adverse_event"].mean()) if not test_df.empty else 0.0

    stats = {
        "total_patients": n_total,
        "train_patients": len(train_pids),
        "val_patients": len(val_pids),
        "test_patients": len(test_pids),
        "total_samples": len(full_dataset),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "adverse_event_rate_overall": round(overall_ae_rate, 4),
        "adverse_event_rate_train": round(train_ae_rate, 4),
        "adverse_event_rate_val": round(val_ae_rate, 4),
        "adverse_event_rate_test": round(test_ae_rate, 4),
    }

    stats_file = output_path / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=== Dataset Builder Summary ===")
    logger.info(f"Train set: {len(train_df):,} samples ({len(train_pids)} patients), AE rate: {train_ae_rate*100:.1f}%")
    logger.info(f"Val set:   {len(val_df):,} samples ({len(val_pids)} patients), AE rate: {val_ae_rate*100:.1f}%")
    logger.info(f"Test set:  {len(test_df):,} samples ({len(test_pids)} patients), AE rate: {test_ae_rate*100:.1f}%")
    logger.info(f"Dataset stats written to {stats_file}")
    logger.info("✅ Dataset build successfully completed!")


if __name__ == "__main__":
    main()
