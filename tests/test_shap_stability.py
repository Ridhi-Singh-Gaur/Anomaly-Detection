"""
Regression and Verification Suite for Member 4 Phase 3 Experiment 3:
TreeSHAP Ensemble Stability Analysis (EXP-PHASE3-SHAP-03).
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
exp_dir = base_dir / "experiments" / "phase3_shap_stability"
paper_file = base_dir / "research_paper" / "complete_ieee_paper.md"
cand_file = ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir) / "outputs" if ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir)).exists() else base_dir / "outputs") / "contextual_anomalies.csv"

TREE_SUBSETS = [25, 50, 100, 150, 200]
PER_TREE_FILES = [f"shap_values_trees_{k:03d}.csv" for k in TREE_SUBSETS]


def test_research_paper_strict_freeze():
    """Verify that the research paper master manuscript has not been touched if present."""
    if not paper_file.exists():
        pytest.skip("Manuscript files excluded from public repository during SocProS double-blind review")
    assert paper_file.stat().st_size == 122229, (
        f"Master paper file size mismatch! Expected 122,229 bytes, got {paper_file.stat().st_size}"
    )


def test_experiment3_deliverables_exist():
    """Verify all 5 per-tree CSVs, summary CSVs, figures, log, and script exist."""
    assert (exp_dir / "run_phase3_shap_stability_experiment.py").exists()
    assert (exp_dir / "execution.log").exists()
    assert (exp_dir / "combined_feature_stability.csv").exists()
    assert (exp_dir / "ensemble_convergence_summary.csv").exists()
    assert (exp_dir / "per_sample_attribution_comparison.csv").exists()
    assert (exp_dir / "feature_ranking_stability.png").exists()
    assert (exp_dir / "ensemble_convergence_metrics.png").exists()

    for fname in PER_TREE_FILES:
        fpath = exp_dir / fname
        assert fpath.exists(), f"Missing per-tree output file: {fname}"


def test_row_counts_and_candidate_alignment():
    """Verify that all per-tree CSVs contain exactly 433 instances (333 candidates + 100 normal)."""
    cand_df = pd.read_csv(cand_file)
    cand_timestamps = cand_df["timestamp"].tolist()
    assert len(cand_timestamps) == 333

    for fname in PER_TREE_FILES:
        df = pd.read_csv(exp_dir / fname)
        assert len(df) == 433, f"{fname} has {len(df)} rows, expected 433"
        # First 333 rows must align with candidate timestamps
        assert df.iloc[:333]["timestamp"].tolist() == cand_timestamps, f"{fname} candidate timestamps misaligned!"
        assert (df.iloc[:333]["sample_type"] == "candidate_anomaly").all()
        assert (df.iloc[333:]["sample_type"] == "normal_observation").all()

    feat_df = pd.read_csv(exp_dir / "combined_feature_stability.csv")
    assert len(feat_df) == 16, f"Expected 16 features, got {len(feat_df)}"


def test_additivity_precision_across_all_subsets():
    """Verify that additivity holds to machine precision (< 1e-12) for every instance and subset."""
    for fname in PER_TREE_FILES:
        df = pd.read_csv(exp_dir / fname)
        assert "additivity_error" in df.columns
        max_err = df["additivity_error"].max()
        assert max_err < 1e-12, f"Additivity error {max_err:.2e} exceeded tolerance in {fname}"


def test_monotonic_convergence_metrics():
    """Verify strictly monotonic convergence of Spearman rho and Cosine similarity as K increases."""
    conv_df = pd.read_csv(exp_dir / "ensemble_convergence_summary.csv")
    rhos = conv_df["spearman_rho"].tolist()
    cosines = conv_df["cosine_similarity"].tolist()

    # Monotonic non-decreasing progression
    for i in range(len(rhos) - 1):
        assert rhos[i] <= rhos[i + 1], f"Spearman rho not non-decreasing: {rhos[i]} > {rhos[i+1]}"
        assert cosines[i] <= cosines[i + 1], f"Cosine similarity not non-decreasing: {cosines[i]} > {cosines[i+1]}"

    # Sample-level monotonic convergence
    sample_df = pd.read_csv(exp_dir / "per_sample_attribution_comparison.csv")
    mean_cos = [sample_df[f"cos_sim_k{k}"].mean() for k in [25, 50, 100, 150]]
    mean_mae = [sample_df[f"mae_k{k}"].mean() for k in [25, 50, 100, 150]]

    for i in range(len(mean_cos) - 1):
        assert mean_cos[i] < mean_cos[i + 1], f"Mean sample cosine not strictly increasing: {mean_cos[i]} >= {mean_cos[i+1]}"
        assert mean_mae[i] > mean_mae[i + 1], f"Mean sample MAE not strictly decreasing: {mean_mae[i]} <= {mean_mae[i+1]}"


def test_top5_feature_set_stability():
    """Verify that Top-5 feature overlap reaches 100% at K=100 and K=150."""
    conv_df = pd.read_csv(exp_dir / "ensemble_convergence_summary.csv")
    k100_overlap = conv_df.loc[conv_df["tree_count"] == 100, "top5_overlap_pct"].values[0]
    k150_overlap = conv_df.loc[conv_df["tree_count"] == 150, "top5_overlap_pct"].values[0]
    k200_overlap = conv_df.loc[conv_df["tree_count"] == 200, "top5_overlap_pct"].values[0]

    assert k100_overlap == 100.0
    assert k150_overlap == 100.0
    assert k200_overlap == 100.0


def test_table9_baseline_exact_reproduction():
    """Verify that the 200-tree subset exactly reproduces the baseline feature rankings."""
    feat_df = pd.read_csv(exp_dir / "combined_feature_stability.csv")
    # Top 5 in reference
    top5_feats = feat_df.sort_values("ref_rank_200")["feature"].tolist()[:5]
    expected_top5 = ["Sub_metering_2", "rolling_std_4", "Global_intensity", "Sub_metering_1", "Global_active_power"]
    assert top5_feats == expected_top5, f"Top 5 mismatch: {top5_feats} vs {expected_top5}"

    # Check top feature attribution value within tolerance
    top_val = feat_df.loc[feat_df["feature"] == "Sub_metering_2", "ref_mean_abs_shap_200"].values[0]
    assert np.isclose(top_val, 0.7644, atol=1e-3)

