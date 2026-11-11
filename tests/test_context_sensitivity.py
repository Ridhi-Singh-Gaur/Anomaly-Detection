"""
Regression and Verification Suite for Member 4 Phase 3 Experiment 2:
Context Matching Sensitivity Analysis (EXP-PHASE3-CONTEXT-02).
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
exp_dir = base_dir / "experiments" / "phase3_context_sensitivity"
paper_file = base_dir / "research_paper" / "complete_ieee_paper.md"
features_file = ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir) / "outputs" if ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir)).exists() else base_dir / "outputs") / "household_15min_features.csv"
cand_file = ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir) / "outputs" if ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir)).exists() else base_dir / "outputs") / "contextual_anomalies.csv"

CONFIG_FILES = [
    "config1_h0_season_off.csv",
    "config2_h0_season_on.csv",
    "config3_h1_season_off.csv",
    "config4_h1_season_on.csv",
    "config5_h2_season_off.csv",
    "config6_h2_season_on.csv",
]


def test_research_paper_strict_freeze():
    """Verify that the research paper master manuscript has not been touched if present."""
    if not paper_file.exists():
        pytest.skip("Manuscript files excluded from public repository during SocProS double-blind review")
    assert paper_file.stat().st_size == 122229, (
        f"Master paper file size mismatch! Expected 122,229 bytes, got {paper_file.stat().st_size}"
    )


def test_experiment2_deliverables_exist():
    """Verify all 6 configuration CSVs, the combined CSV, the script, and the log exist."""
    assert (exp_dir / "run_phase3_sensitivity_experiment.py").exists()
    assert (exp_dir / "execution.log").exists()
    assert (exp_dir / "combined_sensitivity_comparison.csv").exists()

    for fname in CONFIG_FILES:
        fpath = exp_dir / fname
        assert fpath.exists(), f"Configuration output file missing: {fname}"


def test_exact_row_count_and_timestamp_ordering():
    """Verify that all files contain exactly 333 rows aligned with contextual_anomalies.csv."""
    cand_df = pd.read_csv(cand_file)
    cand_timestamps = cand_df["timestamp"].tolist()
    assert len(cand_timestamps) == 333, "Reference candidate list does not have 333 entries."

    comb_df = pd.read_csv(exp_dir / "combined_sensitivity_comparison.csv")
    assert len(comb_df) == 333
    assert comb_df["timestamp"].tolist() == cand_timestamps

    for fname in CONFIG_FILES:
        df = pd.read_csv(exp_dir / fname)
        assert len(df) == 333, f"{fname} has {len(df)} rows, expected 333"
        assert df["timestamp"].tolist() == cand_timestamps, f"{fname} timestamp sequence diverges!"


def test_monotonic_cohort_size_progression():
    """
    Verify fundamental structural properties of the contextual hierarchies:
    1. For any given hour window, season-restricted cohorts must be <= season-unrestricted cohorts.
    2. Widening the hour window (h=0 -> h=1 -> h=2) must monotonically expand the cohort size.
    """
    comb_df = pd.read_csv(exp_dir / "combined_sensitivity_comparison.csv")

    for _, row in comb_df.iterrows():
        # Hour window expansion (Season OFF)
        assert row["c1_cohort_size"] <= row["c3_cohort_size"], "c1 > c3 (h0 > h1 season off) violation!"
        assert row["c3_cohort_size"] <= row["c5_cohort_size"], "c3 > c5 (h1 > h2 season off) violation!"

        # Hour window expansion (Season ON)
        assert row["c2_cohort_size"] <= row["c4_cohort_size"], "c2 > c4 (h0 > h1 season on) violation!"
        assert row["c4_cohort_size"] <= row["c6_cohort_size"], "c4 > c6 (h1 > h2 season on) violation!"

        # Seasonal restriction (Season ON <= Season OFF)
        assert row["c2_cohort_size"] <= row["c1_cohort_size"], "c2 > c1 (season on > season off at h0)!"
        assert row["c4_cohort_size"] <= row["c3_cohort_size"], "c4 > c3 (season on > season off at h1)!"
        assert row["c6_cohort_size"] <= row["c5_cohort_size"], "c6 > c5 (season on > season off at h2)!"


def test_cohort_sufficiency_and_no_missing_power():
    """
    Verify that all candidate anomaly readings have valid telemetry and cohorts >= 50.
    """
    for fname in CONFIG_FILES:
        df = pd.read_csv(exp_dir / fname)
        assert not df["is_power_missing"].any(), f"Missing power detected in {fname}"
        assert not df["is_cohort_insufficient"].any(), f"Cohort insufficient (<50) in {fname}"
        assert (df["cohort_size"] >= 50).all(), f"Cohort size < 50 found in {fname}"


def test_mathematical_consistency_of_metrics():
    """Verify that deviation and z-score match formulas within floating-point precision."""
    for fname in CONFIG_FILES:
        df = pd.read_csv(exp_dir / fname)
        for _, row in df.iterrows():
            p = row["actual_power_kw"]
            med = row["contextual_median_kw"]
            mean = row["contextual_mean_kw"]
            std = row["contextual_std_kw"]
            dev = row["contextual_deviation_kw"]
            z = row["contextual_z_score"]

            expected_dev = abs(p - med)
            expected_z = (p - mean) / max(std, 1e-6)

            assert np.isclose(dev, expected_dev, atol=1e-12), f"Deviation mismatch in {fname}"
            assert np.isclose(z, expected_z, atol=1e-12), f"Z-score mismatch in {fname}"


def test_cross_config_ranges_in_combined():
    """Verify that min/max/range aggregations in combined_sensitivity_comparison.csv are correct."""
    comb_df = pd.read_csv(exp_dir / "combined_sensitivity_comparison.csv")
    for _, row in comb_df.iterrows():
        meds = [row[f"c{i}_median_kw"] for i in range(1, 7)]
        devs = [row[f"c{i}_deviation_kw"] for i in range(1, 7)]
        stds = [row[f"c{i}_std_kw"] for i in range(1, 7)]
        zs = [row[f"c{i}_z_score"] for i in range(1, 7)]

        assert np.isclose(row["median_min_kw"], min(meds), atol=1e-12)
        assert np.isclose(row["median_max_kw"], max(meds), atol=1e-12)
        assert np.isclose(row["median_range_kw"], max(meds) - min(meds), atol=1e-12)

        assert np.isclose(row["deviation_min_kw"], min(devs), atol=1e-12)
        assert np.isclose(row["deviation_max_kw"], max(devs), atol=1e-12)
        assert np.isclose(row["deviation_range_kw"], max(devs) - min(devs), atol=1e-12)

        assert np.isclose(row["std_min_kw"], min(stds), atol=1e-12)
        assert np.isclose(row["std_max_kw"], max(stds), atol=1e-12)
        assert np.isclose(row["std_range_kw"], max(stds) - min(stds), atol=1e-12)

        assert np.isclose(row["z_min"], min(zs), atol=1e-12)
        assert np.isclose(row["z_max"], max(zs), atol=1e-12)
        assert np.isclose(row["z_range"], max(zs) - min(zs), atol=1e-12)

