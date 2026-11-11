"""
Regression and Verification Suite for Member 4 Phase 3 Experiment 4:
Randomized TreeSHAP Stability Validation (EXP-PHASE3-SHAP-04).
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
exp_dir = base_dir / "experiments" / "phase3_shap_randomized_stability"
paper_file = base_dir / "research_paper" / "complete_ieee_paper.md"

TREE_SIZES = [25, 50, 100, 150]
NUM_TRIALS = 20


def test_research_paper_strict_freeze():
    """Verify that the research paper master manuscript has not been touched if present."""
    if not paper_file.exists():
        pytest.skip("Manuscript files excluded from public repository during SocProS double-blind review")
    assert paper_file.stat().st_size == 122229, (
        f"Master paper file size mismatch! Expected 122,229 bytes, got {paper_file.stat().st_size}"
    )


def test_experiment4_deliverables_exist():
    """Verify all deliverables, summary CSVs, figures, log, and script exist."""
    assert (exp_dir / "run_phase3_shap_randomized_stability_experiment.py").exists()
    assert (exp_dir / "execution.log").exists()
    assert (exp_dir / "randomized_runs_summary.csv").exists()
    assert (exp_dir / "randomized_stability_summary.csv").exists()
    assert (exp_dir / "feature_top5_frequency_stability.csv").exists()
    assert (exp_dir / "randomized_ensemble_convergence.png").exists()
    assert (exp_dir / "feature_top5_selection_frequency.png").exists()


def test_run_counts_and_structure():
    """Verify that exactly 80 randomized runs exist (20 trials for each tree size)."""
    df = pd.read_csv(exp_dir / "randomized_runs_summary.csv")
    assert len(df) == 80, f"Expected 80 runs, got {len(df)}"

    for k in TREE_SIZES:
        k_df = df[df["tree_count"] == k]
        assert len(k_df) == NUM_TRIALS, f"Expected {NUM_TRIALS} runs for K={k}, got {len(k_df)}"
        assert k_df["trial"].tolist() == list(range(1, NUM_TRIALS + 1))
        # Unique seeds
        assert len(k_df["seed"].unique()) == NUM_TRIALS


def test_sampling_without_replacement_and_no_duplicates():
    """Verify that tree selections were strictly without replacement and indices in [0, 199]."""
    df = pd.read_csv(exp_dir / "randomized_runs_summary.csv")
    for _, row in df.iterrows():
        k = int(row["tree_count"])
        idx_list = [int(x) for x in row["selected_indices_str"].split(",")]
        assert len(idx_list) == k, f"Tree count mismatch in run {row['run_id']}"
        assert len(set(idx_list)) == k, f"Duplicate tree detected in run {row['run_id']}"
        assert min(idx_list) >= 0 and max(idx_list) < 200, f"Index out of bounds in run {row['run_id']}"


def test_additivity_error_bound():
    """Verify that additivity holds to floating-point machine precision across all 80 runs."""
    df = pd.read_csv(exp_dir / "randomized_runs_summary.csv")
    max_err = df["max_additivity_error"].max()
    assert max_err < 1e-12, f"Additivity error {max_err:.2e} exceeded tolerance!"


def test_monotonic_randomized_convergence():
    """Verify that mean Spearman rho, Cosine, and sample metrics monotonically improve with K."""
    sum_df = pd.read_csv(exp_dir / "randomized_stability_summary.csv")
    rhos = sum_df["spearman_mean"].tolist()
    cosines = sum_df["cosine_mean"].tolist()
    sample_cos = sum_df["sample_cosine_mean"].tolist()
    sample_maes = sum_df["sample_mae_mean"].tolist()
    exact_top5_rates = sum_df["exact_top5_recovery_rate_pct"].tolist()

    for i in range(len(rhos) - 1):
        assert rhos[i] < rhos[i + 1], f"Mean Spearman rho not strictly increasing: {rhos[i]} >= {rhos[i+1]}"
        assert cosines[i] < cosines[i + 1], f"Mean Cosine similarity not strictly increasing: {cosines[i]} >= {cosines[i+1]}"
        assert sample_cos[i] < sample_cos[i + 1], f"Mean sample cosine not strictly increasing: {sample_cos[i]} >= {sample_cos[i+1]}"
        assert sample_maes[i] > sample_maes[i + 1], f"Mean sample MAE not strictly decreasing: {sample_maes[i]} <= {sample_maes[i+1]}"
        assert exact_top5_rates[i] <= exact_top5_rates[i + 1], f"Top5 recovery not non-decreasing: {exact_top5_rates[i]} > {exact_top5_rates[i+1]}"


def test_prefix_baseline_within_randomized_distribution():
    """Verify that Experiment 3 prefix baseline values fall within randomized ranges for K=100 and K=150."""
    sum_df = pd.read_csv(exp_dir / "randomized_stability_summary.csv")
    for k in [100, 150]:
        row = sum_df[sum_df["tree_count"] == k].iloc[0]
        assert row["prefix_rho_within_range"] == True, f"Prefix rho for K={k} outside randomized range!"
        assert row["prefix_cos_within_range"] == True, f"Prefix cosine for K={k} outside randomized range!"

