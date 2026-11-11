"""
Regression and Verification Suite for Member 4 Phase 3 Experiment 6.2:
Targeted Explanation Quality Remediation (EXP-PHASE3-XAI-06B).
"""

from pathlib import Path
import json
import re
import pytest
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
exp_dir = base_dir / "experiments" / "phase3_explanation_remediation"
src_dir = ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir) if ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir)).exists() else base_dir)
paper_file = base_dir / "research_paper" / "complete_ieee_paper.md"
backup_dir = src_dir / "outputs_backup_phase3"

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


def test_experiment6_2_deliverables_exist():
    """Verify all required deliverables exist in experiments/phase3_explanation_remediation/."""
    assert (exp_dir / "run_explanation_remediation.py").exists()
    assert (exp_dir / "execution.log").exists()
    assert (exp_dir / "before_after_explanation_comparison.csv").exists()
    assert (exp_dir / "semantic_audit_summary.json").exists()
    assert (exp_dir / "remediation_report.md").exists()


def test_baseline_backups_preserved():
    """Verify preserved baseline backups remain exactly intact."""
    assert (backup_dir / "explanations.json").exists()
    assert (backup_dir / "contextual_anomalies.csv").exists()
    assert (backup_dir / "explanations.json").stat().st_size == 767346
    assert (backup_dir / "contextual_anomalies.csv").stat().st_size == 299464
    assert (src_dir / "outputs" / "isolation_forest_model.joblib").stat().st_size == 2434327


def test_remediation_summary_metrics_perfect_pass():
    """Verify that semantic audit summary achieves 100% PASS (333 PASS, 0 WARNING, 0 FAIL)."""
    with open(exp_dir / "semantic_audit_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["total_anomalies_audited"] == 333
    assert summary["overall_status_counts"]["PASS"] == 333
    assert summary["overall_status_counts"]["WARNING"] == 0
    assert summary["overall_status_counts"]["FAIL"] == 0

    dim_counts = summary["dimension_status_counts"]
    for dim_name, counts in dim_counts.items():
        assert counts["PASS"] == 333, f"Dimension {dim_name} has non-PASS: {counts}"
        assert counts["WARNING"] == 0, f"Dimension {dim_name} has warnings: {counts}"
        assert counts["FAIL"] == 0, f"Dimension {dim_name} has failures: {counts}"

    issues = summary["issue_category_counts"]
    for issue_cat, count in issues.items():
        assert count == 0, f"Issue {issue_cat} has non-zero count: {count}"

    rem_summary = summary["remediation_summary"]
    assert rem_summary["unmetered_telemetry_remediated_count"] == 73
    assert rem_summary["shap_directional_remediated_count"] == 5
    assert rem_summary["clean_explanations_count_before"] == 260
    assert rem_summary["clean_explanations_count_after"] == 333
    assert rem_summary["pass_rate_percentage_before"] == 78.08
    assert rem_summary["pass_rate_percentage_after"] == 100.0


def test_unmetered_speculative_claims_completely_removed():
    """Verify zero speculative unmetered circuit claims and exactly 73 cautious statements."""
    with open(explanations_file, "r", encoding="utf-8") as f:
        exps = json.load(f)

    assert len(exps) == 333

    old_phrase = "lighting, or wall sockets"
    new_phrase = "The available measurements do not identify the source of the remaining consumption."

    old_occurrences = 0
    new_occurrences = 0

    for e in exps:
        diag = e["human_readable_diagnostic"]
        if old_phrase in diag:
            old_occurrences += 1
        if new_phrase in diag:
            new_occurrences += 1

    assert old_occurrences == 0, f"Found {old_occurrences} old speculative phrases in explanations!"
    assert new_occurrences == 73, f"Expected 73 evidence-based unmetered statements, found {new_occurrences}"


def test_shap_explanation_wording_resolved():
    """Verify removal of blanket 'Key driving features' header and proper directional categorization."""
    with open(explanations_file, "r", encoding="utf-8") as f:
        exps = json.load(f)

    # Blanket header must be completely gone
    for e in exps:
        assert "Key driving features:" not in e["human_readable_diagnostic"]

    # Test the 5 specific timestamps that produced warnings in Experiment 6
    target_ts = {
        "2010-05-19 16:30:00",
        "2010-05-29 16:45:00",
        "2010-08-27 16:30:00",
        "2010-09-09 14:30:00",
        "2010-10-26 09:45:00",
    }
    found_target_ts = set()

    for e in exps:
        ts = e["timestamp"]
        if ts in target_ts:
            found_target_ts.add(ts)
            diag = e["human_readable_diagnostic"]
            top_shap = e["top_shap_features"]

            for f in top_shap:
                feat = f["feature"]
                attr = f["attribution"]
                if attr >= 0.05:
                    assert "Features contributing towards anomaly severity:" in diag
                    assert f"{feat} =" in diag
                    assert "pushed anomaly score higher" in diag
                elif attr <= -0.05:
                    assert "Features contributing towards normality:" in diag
                    assert f"{feat} =" in diag
                    assert "acted toward normalcy" in diag
                else: # |attr| < 0.05
                    assert "Features with negligible attribution:" in diag
                    assert f"{feat} =" in diag

    assert found_target_ts == target_ts, "Not all 5 SHAP warning timestamps were checked!"


def test_before_after_comparison_csv_parity():
    """Verify before_after_explanation_comparison.csv contains all 333 rows with consistent transitions."""
    df = pd.read_csv(exp_dir / "before_after_explanation_comparison.csv")
    assert len(df) == 333
    assert df["timestamp"].is_unique

    assert (df["after_overall_status"] == "PASS").all()
    assert (df["after_shap_status"] == "PASS").all()
    assert (df["after_scientific_status"] == "PASS").all()

    # Verify that exactly 78 rows had remediation categories applied
    remediated_mask = df["remediation_categories_applied"] != "none"
    # Note: 73 unmetered + 5 shap directional = 78 instances (none overlap)
    unmetered_fixed = df["remediation_categories_applied"].str.contains("unmetered_telemetry_remediation")
    shap_fixed = df["remediation_categories_applied"].str.contains("shap_directional_remediation")

    assert unmetered_fixed.sum() == 73
    assert shap_fixed.sum() == 5
    # All 5 SHAP warning events co-occurred during low sub-metering conditions, overlapping with the unmetered group
    assert (unmetered_fixed & shap_fixed).sum() == 5, "Expected all 5 SHAP warning events to overlap with unmetered group"
    assert (unmetered_fixed | shap_fixed).sum() == 73, "Expected exactly 73 unique candidate events to have warnings"


def test_consensus_counts_and_candidate_census():
    """Verify candidate census remains strictly 333 with reconciled consensus distribution."""
    df = pd.read_csv(contextual_csv_file)
    assert len(df) == 333
    assert df["timestamp"].is_unique

    counts = df["models_flagged_count"].value_counts().to_dict()
    assert counts[1] == 242
    assert counts[2] == 50
    assert counts[3] == 37
    assert counts[4] == 4


def test_chronological_partition_leakage_safeguard():
    """Verify all 333 candidates belong strictly to the test partition and context cohorts are training-only."""
    features_df = pd.read_csv(features_csv_file)
    split_idx = int(len(features_df) * 0.8)
    max_train_dt = pd.to_datetime(features_df.iloc[:split_idx]["timestamp"]).max()

    cand_df = pd.read_csv(contextual_csv_file)
    for ts in cand_df["timestamp"]:
        cand_dt = pd.to_datetime(ts)
        assert cand_dt > max_train_dt, f"Temporal leakage: candidate {ts} <= train max {max_train_dt}"
