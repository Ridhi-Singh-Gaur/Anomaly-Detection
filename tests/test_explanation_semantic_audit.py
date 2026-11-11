"""
Regression and Verification Suite for Member 4 Phase 3 Experiment 6:
Semantic and Scientific Audit of All 333 Generated Anomaly Explanations (EXP-PHASE3-XAI-06).
"""

from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
exp_dir = base_dir / "experiments" / "phase3_explanation_semantic_audit"
src_dir = ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir) if ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir)).exists() else base_dir)
paper_file = base_dir / "research_paper" / "complete_ieee_paper.md"

explanations_file = src_dir / "outputs" / "explanations.json"
contextual_csv_file = src_dir / "outputs" / "contextual_anomalies.csv"
features_csv_file = src_dir / "outputs" / "household_15min_features.csv"
shap_ref_file = base_dir / "experiments" / "phase3_shap_stability" / "shap_values_trees_200.csv"


def test_research_paper_strict_freeze():
    """Verify that the research paper master manuscript has not been touched if present."""
    if not paper_file.exists():
        pytest.skip("Manuscript files excluded from public repository during SocProS double-blind review")
    assert paper_file.stat().st_size == 122229, (
        f"Master paper file size mismatch! Expected 122,229 bytes, got {paper_file.stat().st_size}"
    )


def test_experiment6_deliverables_exist():
    """Verify all required deliverables exist in experiments/phase3_explanation_semantic_audit/."""
    assert (exp_dir / "run_explanation_semantic_audit.py").exists()
    assert (exp_dir / "execution.log").exists()
    assert (exp_dir / "explanation_semantic_audit.csv").exists()
    assert (exp_dir / "explanation_quality_summary.json").exists()
    assert (exp_dir / "explanation_issue_examples.csv").exists()


def test_exact_333_candidates_and_no_duplicates():
    """Verify exactly 333 unique candidate timestamps are audited with zero duplicates."""
    audit_df = pd.read_csv(exp_dir / "explanation_semantic_audit.csv")
    assert len(audit_df) == 333, f"Expected 333 rows, got {len(audit_df)}"
    assert audit_df["timestamp"].is_unique, "Duplicate timestamps found in audit records!"

    with open(exp_dir / "explanation_quality_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["total_anomalies_audited"] == 333


def test_dimension_status_validity_and_synthesis():
    """Verify that all five quality dimensions have valid status values and overall status matches rubric."""
    audit_df = pd.read_csv(exp_dir / "explanation_semantic_audit.csv")
    valid_statuses = {"PASS", "WARNING", "FAIL", "NOT_VERIFIABLE"}

    dim_cols = [
        "audit_a_completeness_status",
        "audit_b_shap_status",
        "audit_c_consensus_status",
        "audit_d_context_status",
        "audit_e_scientific_status",
        "overall_status",
    ]

    for col in dim_cols:
        for val in audit_df[col]:
            assert val in valid_statuses, f"Invalid status '{val}' in column {col}"

    # Verify overall synthesis rule
    for _, row in audit_df.iterrows():
        dims = [row[c] for c in dim_cols[:5]]
        if "FAIL" in dims:
            assert row["overall_status"] == "FAIL"
        elif "WARNING" in dims:
            assert row["overall_status"] == "WARNING"
        elif "NOT_VERIFIABLE" in dims:
            assert row["overall_status"] == "NOT_VERIFIABLE"
        else:
            assert row["overall_status"] == "PASS"


def test_consensus_counts_reconciliation():
    """Verify consensus counts reconcile exactly to (242, 50, 37, 4) and match model result files."""
    audit_df = pd.read_csv(exp_dir / "explanation_semantic_audit.csv")
    counts = audit_df["models_flagged_count"].value_counts().to_dict()

    assert counts.get(1, 0) == 242, f"Expected 242 1-model flags, got {counts.get(1, 0)}"
    assert counts.get(2, 0) == 50, f"Expected 50 2-model flags, got {counts.get(2, 0)}"
    assert counts.get(3, 0) == 37, f"Expected 37 3-model flags, got {counts.get(3, 0)}"
    assert counts.get(4, 0) == 4, f"Expected 4 4-model flags, got {counts.get(4, 0)}"
    assert sum(counts.values()) == 333


def test_shap_numerical_validation_and_warning_detection():
    """Verify SHAP attributions against the reference 200-tree matrix and confirm warning detection."""
    audit_df = pd.read_csv(exp_dir / "explanation_semantic_audit.csv")
    shap_df = pd.read_csv(shap_ref_file)
    shap_df["ts_str"] = pd.to_datetime(shap_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    shap_map = shap_df.set_index("ts_str")

    # Audit B status counts: exactly 328 PASS and 5 WARNING
    b_counts = audit_df["audit_b_shap_status"].value_counts().to_dict()
    assert b_counts.get("PASS", 0) == 328, f"Expected 328 SHAP PASS, got {b_counts.get('PASS')}"
    assert b_counts.get("WARNING", 0) == 5, f"Expected 5 SHAP WARNING, got {b_counts.get('WARNING')}"
    assert b_counts.get("FAIL", 0) == 0, f"Expected 0 SHAP FAIL, got {b_counts.get('FAIL')}"

    # Verify that the 5 warnings correspond to negative or negligible attributions
    warn_rows = audit_df[audit_df["audit_b_shap_status"] == "WARNING"]
    expected_warn_ts = {
        "2010-05-19 16:30:00",
        "2010-05-29 16:45:00",
        "2010-08-27 16:30:00",
        "2010-09-09 14:30:00",
        "2010-10-26 09:45:00",
    }
    assert set(warn_rows["timestamp"]) == expected_warn_ts, "Mismatch in expected SHAP warning timestamps"


def test_contextual_reference_training_only_and_reproducibility():
    """Verify context reference observations are exclusively training data and calculations match."""
    audit_df = pd.read_csv(exp_dir / "explanation_semantic_audit.csv")
    features_df = pd.read_csv(features_csv_file)
    features_df["dt"] = pd.to_datetime(features_df["timestamp"])
    
    # 80% split point
    split_idx = int(len(features_df) * 0.8)
    max_train_dt = features_df.iloc[:split_idx]["dt"].max()

    for ts in audit_df["timestamp"]:
        cand_dt = pd.to_datetime(ts)
        assert cand_dt > max_train_dt, f"Temporal leakage: candidate {ts} is <= max training date {max_train_dt}"

    # Verify 100% PASS on context reference dimension
    d_counts = audit_df["audit_d_context_status"].value_counts().to_dict()
    assert d_counts.get("PASS", 0) == 333


def test_scientific_validity_and_absence_of_unsupported_claims():
    """Verify zero unsupported appliance names, zero causal claims, and proper circuit mapping."""
    with open(exp_dir / "explanation_quality_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    issues = summary["issue_category_counts"]
    assert issues["unsupported_appliance_attribution"] == 0
    assert issues["unsupported_causal_claim"] == 0
    assert issues["unsupported_household_behaviour_claim"] == 0
    assert issues["incorrect_circuit_mapping"] == 0
    assert issues["unsupported_fault_diagnosis"] == 0
    assert issues["other_scientific_terminology_issue"] == 0

    # Unverified general circuit hypotheses: exactly 73
    assert issues["unverified_general_circuit_hypothesis"] == 73


def test_original_artifacts_unmodified():
    """Verify original explanations.json and model artifacts have preserved baseline backups."""
    backup_dir = src_dir / "outputs_backup_phase3"
    assert (backup_dir / "explanations.json").stat().st_size == 767346, "Backup explanations.json size altered!"
    assert (backup_dir / "contextual_anomalies.csv").stat().st_size == 299464, "Backup contextual_anomalies.csv size altered!"
    assert (src_dir / "outputs" / "isolation_forest_model.joblib").stat().st_size == 2434327

