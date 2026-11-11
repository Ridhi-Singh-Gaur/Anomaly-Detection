"""
Regression and Verification Suite for Member 4 Phase 3 Experiment 6.1:
Chronological Partition and Audit Integrity Reconciliation (EXP-PHASE3-XAI-06A).
"""

from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
rec_dir = base_dir / "experiments" / "phase3_explanation_semantic_audit" / "reconciliation"
src_dir = ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir) if ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir)).exists() else base_dir)
paper_file = base_dir / "research_paper" / "complete_ieee_paper.md"

features_csv_file = src_dir / "outputs" / "household_15min_features.csv"
explanations_file = src_dir / "outputs" / "explanations.json"
contextual_csv_file = src_dir / "outputs" / "contextual_anomalies.csv"


def test_research_paper_strict_freeze():
    """Verify that the research paper master manuscript has not been touched if present."""
    if not paper_file.exists():
        pytest.skip("Manuscript files excluded from public repository during SocProS double-blind review")
    assert paper_file.stat().st_size == 122229, (
        f"Master paper file size mismatch! Expected 122,229 bytes, got {paper_file.stat().st_size}"
    )


def test_reconciliation_deliverables_exist():
    """Verify all required deliverables exist in reconciliation/."""
    assert (rec_dir / "run_partition_reconciliation.py").exists()
    assert (rec_dir / "execution.log").exists()
    assert (rec_dir / "candidate_partition_membership.csv").exists()
    assert (rec_dir / "partition_integrity_summary.json").exists()
    assert (rec_dir / "partition_reconciliation_report.md").exists()


def test_actual_original_partition_sizes_and_indices():
    """Verify exact partition observation counts and slice indices."""
    df = pd.read_csv(features_csv_file)
    assert len(df) == 136684, f"Expected 136,684 total observations, got {len(df)}"

    split_idx = int(len(df) * 0.8)
    assert split_idx == 109347, f"Expected split index 109,347, got {split_idx}"

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    assert len(train_df) == 109347
    assert len(test_df) == 27337
    assert len(train_df) + len(test_df) == len(df)


def test_exact_chronological_boundary_and_no_overlap():
    """Verify exact start and end timestamps and assert zero temporal overlap."""
    df = pd.read_csv(features_csv_file)
    df["dt"] = pd.to_datetime(df["timestamp"])
    assert df["dt"].is_monotonic_increasing, "Dataset timestamps are not monotonically increasing!"

    split_idx = 109347
    train_start = df["dt"].iloc[0]
    train_end = df["dt"].iloc[split_idx - 1]
    test_start = df["dt"].iloc[split_idx]
    test_end = df["dt"].iloc[-1]

    assert str(train_start) == "2006-12-16 17:15:00"
    assert str(train_end) == "2010-02-05 05:15:00"
    assert str(test_start) == "2010-02-05 05:30:00"
    assert str(test_end) == "2010-11-26 21:00:00"

    # Exact 15-minute gap
    gap_min = (test_start - train_end).total_seconds() / 60.0
    assert gap_min == 15.0, f"Expected 15.0 min boundary gap, got {gap_min}"
    assert train_end < test_start


def test_candidate_membership_all_in_test_partition():
    """Verify all 333 candidate anomalies belong to the original test partition."""
    member_df = pd.read_csv(rec_dir / "candidate_partition_membership.csv")
    assert len(member_df) == 333
    assert member_df["is_in_test_partition"].all(), "Not all candidates belong to the test partition!"
    assert not member_df["is_in_training_partition"].any(), "Candidates detected in training partition!"

    # Check temporal gap from train end (all strictly positive)
    assert (member_df["temporal_gap_from_train_end_hours"] > 0).all()

    # Earliest candidate
    first_row = member_df.iloc[0]
    assert first_row["timestamp"] == "2010-02-05 06:45:00"
    assert first_row["dataset_global_index"] == 109352
    assert first_row["test_slice_relative_index"] == 5

    # Latest candidate
    last_row = member_df.iloc[-1]
    assert last_row["timestamp"] == "2010-11-26 06:45:00"
    assert last_row["dataset_global_index"] == 136626
    assert last_row["test_slice_relative_index"] == 27279


def test_historical_context_references_training_only():
    """Verify that all historical twin context cohorts derive exclusively from training."""
    member_df = pd.read_csv(rec_dir / "candidate_partition_membership.csv")
    assert member_df["historical_cohort_training_only"].all()

    with open(rec_dir / "partition_integrity_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["candidate_anomalies"]["membership_in_training_partition"] == 0
    assert summary["candidate_anomalies"]["membership_in_test_partition"] == 333


def test_original_artifacts_unmodified_safeguard():
    """Verify that all original production artifacts remain untouched in preserved backups."""
    backup_dir = src_dir / "outputs_backup_phase3"
    assert (src_dir / "outputs" / "household_15min_features.csv").stat().st_size in (32628104, 32764789)
    assert (backup_dir / "explanations.json").stat().st_size == 767346
    assert (backup_dir / "contextual_anomalies.csv").stat().st_size == 299464
    assert (src_dir / "outputs" / "isolation_forest_model.joblib").stat().st_size == 2434327

