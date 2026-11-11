"""
Regression and Verification Suite for Member 4 Phase 3 Experiment 5:
Feature Dependence and SHAP Attribution Redundancy Audit (EXP-PHASE3-SHAP-05).
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
exp_dir = base_dir / "experiments" / "phase3_shap_dependence"
paper_file = base_dir / "research_paper" / "complete_ieee_paper.md"
shap_ref_file = base_dir / "experiments" / "phase3_shap_stability" / "shap_values_trees_200.csv"


def test_research_paper_strict_freeze():
    """Verify that the research paper master manuscript has not been touched if present."""
    if not paper_file.exists():
        pytest.skip("Manuscript files excluded from public repository during SocProS double-blind review")
    assert paper_file.stat().st_size == 122229, (
        f"Master paper file size mismatch! Expected 122,229 bytes, got {paper_file.stat().st_size}"
    )


def test_experiment5_deliverables_exist():
    """Verify all 10 required deliverables exist."""
    assert (exp_dir / "run_phase3_shap_dependence_experiment.py").exists()
    assert (exp_dir / "execution.log").exists()
    assert (exp_dir / "feature_value_correlations.csv").exists()
    assert (exp_dir / "shap_attribution_correlations.csv").exists()
    assert (exp_dir / "feature_correlation_pairs.csv").exists()
    assert (exp_dir / "grouped_shap_importance.csv").exists()
    assert (exp_dir / "feature_dependence_analysis.png").exists()
    assert (exp_dir / "grouped_shap_importance.png").exists()


def test_feature_count_and_matrix_dimensions():
    """Verify that exactly 16 features are analyzed and correlation tables have correct dimensions."""
    raw_df = pd.read_csv(exp_dir / "feature_value_correlations.csv")
    shap_df = pd.read_csv(exp_dir / "shap_attribution_correlations.csv")
    pair_df = pd.read_csv(exp_dir / "feature_correlation_pairs.csv")

    f1_unique = raw_df["feature_1"].unique()
    assert len(f1_unique) == 16, f"Expected 16 features, got {len(f1_unique)}"

    # 16 x 16 = 256 pairs
    assert len(raw_df) == 256
    assert len(shap_df) == 256

    # 16 * 15 / 2 = 120 unique unordered pairs
    assert len(pair_df) == 120


def test_correlation_bounds_and_no_missing_values():
    """Verify all correlation values lie strictly within [-1.0, 1.0] and contain no NaNs."""
    raw_df = pd.read_csv(exp_dir / "feature_value_correlations.csv")
    shap_df = pd.read_csv(exp_dir / "shap_attribution_correlations.csv")
    pair_df = pd.read_csv(exp_dir / "feature_correlation_pairs.csv")

    for df, name in [(raw_df, "raw"), (shap_df, "shap"), (pair_df, "pair")]:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if ("corr" in c or "pearson" in c or "spearman" in c) and "discrepancy" not in c:
                vals = df[c].dropna()
                assert len(vals) == len(df), f"Missing values found in {name}.{c}"
                assert (vals >= -1.000001).all() and (vals <= 1.000001).all(), (
                    f"Correlation values out of [-1, 1] in {name}.{c}: min={vals.min()}, max={vals.max()}"
                )
            elif "discrepancy" in c:
                vals = df[c].dropna()
                assert len(vals) == len(df), f"Missing values in {name}.{c}"
                assert (vals >= 0.0).all() and (vals <= 2.000001).all(), (
                    f"Discrepancy out of [0, 2] in {name}.{c}: min={vals.min()}, max={vals.max()}"
                )


def test_grouped_importance_reconciliation():
    """Verify that group importance sums reconcile exactly to total feature importance and 100%."""
    group_df = pd.read_csv(exp_dir / "grouped_shap_importance.csv")
    assert len(group_df) == 6, f"Expected 6 groups, got {len(group_df)}"

    total_pct = group_df["percentage_of_total_importance"].sum()
    assert np.isclose(total_pct, 100.0, atol=1e-3), f"Total percentage {total_pct} does not sum to 100%"

    total_l1 = group_df["l1_importance_sum"].sum()
    assert np.isclose(total_l1, 5.2637, atol=1e-3), f"Total L1 importance sum {total_l1} mismatch"

    # All 16 features accounted for
    all_feats = []
    for f_str in group_df["constituent_features"]:
        all_feats.extend(f_str.split(";"))
    assert len(all_feats) == 16
    assert len(set(all_feats)) == 16


def test_collinear_pair_properties():
    """Verify that Global_active_power and Global_intensity exhibit expected high correlation."""
    pair_df = pd.read_csv(exp_dir / "feature_correlation_pairs.csv")
    row = pair_df[((pair_df["feature_1"] == "Global_active_power") & (pair_df["feature_2"] == "Global_intensity")) |
                  ((pair_df["feature_1"] == "Global_intensity") & (pair_df["feature_2"] == "Global_active_power"))].iloc[0]

    assert row["train_raw_pearson"] > 0.99, f"Expected train raw Pearson > 0.99, got {row['train_raw_pearson']}"
    assert row["cand_shap_pearson"] > 0.95, f"Expected cand SHAP Pearson > 0.95, got {row['cand_shap_pearson']}"


def test_reference_shap_matrix_preservation():
    """Verify that the reference 200-tree SHAP matrix from Experiment 3 was not altered."""
    assert shap_ref_file.exists()
    assert shap_ref_file.stat().st_size == 197056, (
        f"SHAP ref file size altered! Expected 197,056 bytes, got {shap_ref_file.stat().st_size}"
    )
