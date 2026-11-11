"""
Phase 3: Experimental Validation of Context-Aware XAI
Experiment 3: TreeSHAP Ensemble Stability Analysis
Experiment ID: EXP-PHASE3-SHAP-03

Evaluates the structural stability and convergence of TreeSHAP explanations
on the fitted 200-tree Isolation Forest across nested tree subsets:
  - 25 trees  (12.5% ensemble)
  - 50 trees  (25.0% ensemble)
  - 100 trees (50.0% ensemble)
  - 150 trees (75.0% ensemble)
  - 200 trees (100.0% full reference ensemble)

Samples evaluated:
  - 333 candidate anomalies (multi-model consensus pool)
  - 100 randomly sampled normal test instances (random_state=42)
  - Total evaluation pool: 433 instances

Evaluates:
  1. Mean absolute SHAP values per feature
  2. Feature rankings across all 16 features
  3. Spearman rank correlation against 200-tree reference
  4. Cosine similarity of global attribution vectors
  5. Top-five feature overlap and stability
  6. Per-sample attribution similarity distributions
  7. Exact additivity error against subset model explained function
"""

from pathlib import Path
import sys
import os
import copy
import logging
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble._iforest import _average_path_length

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup paths
base_dir = Path(__file__).resolve().parent.parent.parent
src_dir = ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir) if ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir)).exists() else base_dir)
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import shap

TREE_SUBSETS = [25, 50, 100, 150, 200]


def run_shap_stability_experiment():
    exp_dir = Path(__file__).resolve().parent
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = exp_dir / "execution.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("EXP-PHASE3-SHAP-03")

    logger.info("=" * 76)
    logger.info("MEMBER 4 - PHASE 3: TREESHAP ENSEMBLE STABILITY ANALYSIS")
    logger.info("Experiment ID: EXP-PHASE3-SHAP-03")
    logger.info("=" * 76)

    # 1. Load Model and Features
    model_path = src_dir / "outputs" / "isolation_forest_model.joblib"
    saved = joblib.load(model_path)
    model = saved["model"]
    scaler = saved["scaler"]
    features = list(saved["feature_columns"])
    medians = saved["training_medians"]
    logger.info(f"Loaded Isolation Forest model: {len(model.estimators_)} trees, {len(features)} features")

    # 2. Load Evaluation Data (333 Candidates + 100 Normal)
    cand_csv = src_dir / "outputs" / "contextual_anomalies.csv"
    feat_csv = src_dir / "outputs" / "household_15min_features.csv"
    candidates_df = pd.read_csv(cand_csv)
    feat_df = pd.read_csv(feat_csv)
    feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"])
    candidates_df["timestamp"] = pd.to_datetime(candidates_df["timestamp"])

    cand_merged = pd.merge(candidates_df[["timestamp"]], feat_df, on="timestamp", how="left")
    cand_merged["sample_type"] = "candidate_anomaly"
    assert len(cand_merged) == 333, f"Expected 333 candidates, got {len(cand_merged)}"

    test_start = pd.to_datetime("2010-02-05 05:30:00")
    test_df = feat_df[feat_df["timestamp"] >= test_start].copy()
    normal_df = test_df[~test_df["timestamp"].isin(candidates_df["timestamp"])].sample(n=100, random_state=42).copy()
    normal_df["sample_type"] = "normal_observation"

    combined_eval = pd.concat([cand_merged, normal_df], ignore_index=True)
    logger.info(f"Evaluation sample assembled: {len(combined_eval)} instances (333 candidates + 100 normal)")

    # 3. Preprocess Features
    x_raw = combined_eval[features].replace([np.inf, -np.inf], np.nan).fillna(medians)
    x_scaled = scaler.transform(x_raw)
    x_cand_scaled = x_scaled[:333]

    # Containers for results
    subset_explainers = {}
    subset_shap_values = {}
    subset_shap_anom = {}
    subset_expected_values = {}
    subset_additivity_errors = {}
    subset_mean_abs_cand = {}
    subset_ranks_cand = {}
    subset_top5_cand = {}

    # 4. Evaluate Each Tree Subset
    for k in TREE_SUBSETS:
        logger.info(f"\n--- Evaluating Subset K={k} trees ---")
        sub_model = copy.deepcopy(model)
        sub_model.estimators_ = sub_model.estimators_[:k]
        sub_model.estimators_features_ = sub_model.estimators_features_[:k]
        sub_model.n_estimators = k

        explainer = shap.TreeExplainer(sub_model)
        exp_val = float(explainer.expected_value[0])
        subset_explainers[k] = explainer
        subset_expected_values[k] = exp_val

        # Compute SHAP for all 433 instances
        shap_vals = explainer.shap_values(x_scaled)
        shap_anom = -shap_vals  # Inverted SHAP for anomaly severity

        subset_shap_values[k] = shap_vals
        subset_shap_anom[k] = shap_anom

        # Verify additivity against direct tree path length on subset
        reconstructed = exp_val + np.sum(shap_vals, axis=1)

        paths = []
        for est in sub_model.estimators_:
            tree = est.tree_
            node_indicators = est.decision_path(x_scaled).toarray()
            leaf_ids = est.apply(x_scaled)
            depths = node_indicators.sum(axis=1) - 1
            leaf_samples = tree.n_node_samples[leaf_ids]
            c_leaf = _average_path_length(leaf_samples)
            paths.append(depths + c_leaf)
        direct_avg_path = np.mean(paths, axis=0)

        add_errs = np.abs(reconstructed - direct_avg_path)
        max_err = float(np.max(add_errs))
        mean_err = float(np.mean(add_errs))
        subset_additivity_errors[k] = add_errs

        logger.info(f"  Base value E[h(x)]: {exp_val:.4f}")
        logger.info(f"  Additivity Max Error: {max_err:.4e} | Mean Error: {mean_err:.4e}")
        assert max_err < 1e-12, f"Additivity error {max_err} exceeds tolerance for K={k}!"

        # Candidate-only metrics (first 333 rows)
        shap_anom_cand = shap_anom[:333]
        mean_abs = np.mean(np.abs(shap_anom_cand), axis=0)
        order = np.argsort(-mean_abs)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(features) + 1)

        subset_mean_abs_cand[k] = mean_abs
        subset_ranks_cand[k] = ranks
        top5 = [features[i] for i in order[:5]]
        subset_top5_cand[k] = top5
        logger.info(f"  Top 5 Features: {top5}")

        # Export per-configuration CSV (all 433 instances)
        cfg_records = []
        for idx, row in combined_eval.iterrows():
            rec = {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "sample_type": row["sample_type"],
                "actual_power_kw": float(row["Global_active_power"]),
                "base_value_path_length": exp_val,
                "reconstructed_path_length": float(reconstructed[idx]),
                "direct_avg_path_length": float(direct_avg_path[idx]),
                "additivity_error": float(add_errs[idx]),
            }
            # Append inverted SHAP values
            for f_idx, feat in enumerate(features):
                rec[f"shap_{feat}"] = float(shap_anom[idx, f_idx])
            cfg_records.append(rec)

        cfg_df = pd.DataFrame(cfg_records)
        cfg_csv = exp_dir / f"shap_values_trees_{k:03d}.csv"
        cfg_df.to_csv(cfg_csv, index=False)
        logger.info(f"  Saved per-configuration CSV ({len(cfg_df)} rows): {cfg_csv.name}")

    # 5. Build Combined Feature Stability CSV
    logger.info("\nBuilding combined_feature_stability.csv...")
    feat_records = []
    ref_ranks = subset_ranks_cand[200]
    ref_means = subset_mean_abs_cand[200]

    for f_idx, feat in enumerate(features):
        row = {
            "feature": feat,
            "ref_rank_200": int(ref_ranks[f_idx]),
            "ref_mean_abs_shap_200": float(ref_means[f_idx]),
        }
        for k in TREE_SUBSETS:
            row[f"rank_k{k}"] = int(subset_ranks_cand[k][f_idx])
            row[f"mean_abs_shap_k{k}"] = float(subset_mean_abs_cand[k][f_idx])
            row[f"rank_diff_k{k}_vs_ref"] = int(subset_ranks_cand[k][f_idx] - ref_ranks[f_idx])
            row[f"abs_diff_k{k}_vs_ref"] = float(subset_mean_abs_cand[k][f_idx] - ref_means[f_idx])

        feat_records.append(row)

    feat_df = pd.DataFrame(feat_records).sort_values("ref_rank_200").reset_index(drop=True)
    feat_csv = exp_dir / "combined_feature_stability.csv"
    feat_df.to_csv(feat_csv, index=False)
    logger.info(f"Saved feature stability CSV ({len(feat_df)} features): {feat_csv.name}")

    # 6. Global Ensemble Convergence Metrics (Table Summary)
    logger.info("\n=== GLOBAL ENSEMBLE CONVERGENCE METRICS ===")
    ref_top5 = set(subset_top5_cand[200])
    convergence_summary = []

    for k in TREE_SUBSETS:
        rho, p_val = spearmanr(subset_ranks_cand[k], ref_ranks)
        cos = float(cosine_similarity(
            subset_mean_abs_cand[k].reshape(1, -1),
            ref_means.reshape(1, -1)
        )[0, 0])
        k_top5 = set(subset_top5_cand[k])
        jaccard = len(k_top5.intersection(ref_top5)) / len(k_top5.union(ref_top5))
        overlap_pct = (len(k_top5.intersection(ref_top5)) / 5.0) * 100.0
        max_err = float(np.max(subset_additivity_errors[k]))

        logger.info(f"K={k:3d} | Spearman Rho={rho:.4f} | Cosine Sim={cos:.6f} | Top5 Overlap={overlap_pct:.0f}% (Jaccard={jaccard:.2f})")
        convergence_summary.append({
            "tree_count": k,
            "spearman_rho": float(rho),
            "cosine_similarity": float(cos),
            "top5_overlap_pct": float(overlap_pct),
            "top5_jaccard": float(jaccard),
            "expected_path_length": float(subset_expected_values[k]),
            "max_additivity_error": max_err,
        })

    conv_df = pd.DataFrame(convergence_summary)
    conv_csv = exp_dir / "ensemble_convergence_summary.csv"
    conv_df.to_csv(conv_csv, index=False)

    # 7. Per-Sample Attribution Similarity Evaluation
    logger.info("\nEvaluating per-sample attribution similarities against K=200 reference...")
    ref_shap_all = subset_shap_anom[200]  # (433, 16)
    sample_records = []

    for idx, row in combined_eval.iterrows():
        ts_str = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        rec = {
            "timestamp": ts_str,
            "sample_type": row["sample_type"],
            "actual_power_kw": float(row["Global_active_power"]),
        }
        ref_vec = ref_shap_all[idx].reshape(1, -1)

        for k in [25, 50, 100, 150]:
            k_vec = subset_shap_anom[k][idx].reshape(1, -1)
            # Cosine similarity
            c_sim = float(cosine_similarity(k_vec, ref_vec)[0, 0])
            # Pearson correlation
            p_corr, _ = pearsonr(k_vec[0], ref_vec[0])
            # Spearman correlation
            s_rho, _ = spearmanr(k_vec[0], ref_vec[0])
            # Mean absolute difference
            mae = float(np.mean(np.abs(k_vec - ref_vec)))
            # L2 distance
            l2 = float(np.linalg.norm(k_vec - ref_vec))

            rec[f"cos_sim_k{k}"] = c_sim
            rec[f"pearson_k{k}"] = float(p_corr)
            rec[f"spearman_k{k}"] = float(s_rho)
            rec[f"mae_k{k}"] = mae
            rec[f"l2_dist_k{k}"] = l2

        sample_records.append(rec)

    sample_df = pd.DataFrame(sample_records)
    sample_csv = exp_dir / "per_sample_attribution_comparison.csv"
    sample_df.to_csv(sample_csv, index=False)
    logger.info(f"Saved per-sample comparison CSV ({len(sample_df)} rows): {sample_csv.name}")

    # Log per-sample cosine similarity percentiles
    logger.info("\nPer-Sample Cosine Similarity Distributions against 200-tree baseline:")
    for k in [25, 50, 100, 150]:
        c_col = f"cos_sim_k{k}"
        logger.info(f"  K={k:3d}: Min={sample_df[c_col].min():.4f}, Q25={sample_df[c_col].quantile(0.25):.4f}, Median={sample_df[c_col].median():.4f}, Mean={sample_df[c_col].mean():.4f}, Max={sample_df[c_col].max():.4f}")

    # 8. Generate Publication-Quality Visualizations
    logger.info("\nGenerating publication figures...")

    # Figure 1: Feature Ranking Stability across Tree Subsets (Slope/Bump Chart)
    fig1, ax1 = plt.subplots(figsize=(10, 7), dpi=300)
    top_indices = np.argsort(ref_ranks)[:8]  # Top 8 features
    other_indices = np.argsort(ref_ranks)[8:]

    # Plot lower rank features faintly
    for idx in other_indices:
        r_vals = [subset_ranks_cand[k][idx] for k in TREE_SUBSETS]
        ax1.plot(TREE_SUBSETS, r_vals, color="#cccccc", linewidth=1.2, linestyle="--", alpha=0.7)

    # Plot top 8 features with distinct vibrant colors
    cmap = plt.cm.tab10
    for color_idx, idx in enumerate(top_indices):
        feat_name = features[idx]
        r_vals = [subset_ranks_cand[k][idx] for k in TREE_SUBSETS]
        line = ax1.plot(TREE_SUBSETS, r_vals, marker="o", linewidth=2.4, color=cmap(color_idx), label=f"{ref_ranks[idx]}. {feat_name}")
        # Annotate end point
        ax1.text(202, r_vals[-1], f" {feat_name}", va="center", fontsize=9, fontweight="bold", color=cmap(color_idx))

    ax1.set_title("TreeSHAP Feature Ranking Stability Across Ensemble Subsets", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Number of Estimator Trees in Ensemble (K)", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Global Feature Rank (1 = Most Influential)", fontsize=11, fontweight="semibold")
    ax1.set_xticks(TREE_SUBSETS)
    ax1.set_yticks(range(1, 17))
    ax1.invert_yaxis()  # Rank 1 at top
    ax1.set_xlim(20, 245)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    fig1_path = exp_dir / "feature_ranking_stability.png"
    plt.savefig(fig1_path)
    plt.close(fig1)
    logger.info(f"Saved Figure 1: {fig1_path.name}")

    # Figure 2: Multi-Panel Ensemble Convergence Metrics
    fig2, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(13, 10), dpi=300)

    # Panel A: Correlation and Cosine Similarity Curves
    ax_a.plot(conv_df["tree_count"], conv_df["spearman_rho"], marker="s", color="#1f77b4", linewidth=2, label="Spearman Rank Correlation (ρ)")
    ax_a.plot(conv_df["tree_count"], conv_df["cosine_similarity"], marker="^", color="#2ca02c", linewidth=2, label="Cosine Similarity")
    ax_a.set_title("(A) Global Attribution Vector Alignment vs. 200 Trees", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Ensemble Size (Trees)", fontsize=10)
    ax_a.set_ylabel("Similarity Score", fontsize=10)
    ax_a.set_xticks(TREE_SUBSETS)
    ax_a.set_ylim(0.85, 1.01)
    ax_a.grid(True, linestyle=":", alpha=0.5)
    ax_a.legend(loc="lower right", fontsize=9)

    # Panel B: Per-Sample Cosine Similarity Distribution (Boxplots)
    box_data = [sample_df[f"cos_sim_k{k}"] for k in [25, 50, 100, 150]]
    box = ax_b.boxplot(box_data, patch_artist=True)
    ax_b.set_xticks(range(1, 5))
    ax_b.set_xticklabels(["K=25", "K=50", "K=100", "K=150"])
    for patch in box["boxes"]:
        patch.set_facecolor("#e0f3f8")
        patch.set_edgecolor("#2b83ba")
    ax_b.set_title("(B) Per-Sample Attribution Cosine Similarity to Baseline", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Tree Subset Configuration", fontsize=10)
    ax_b.set_ylabel("Cosine Similarity per Instance", fontsize=10)
    ax_b.set_ylim(0.80, 1.01)
    ax_b.grid(True, linestyle=":", alpha=0.5)

    # Panel C: Top-5 Feature Magnitudes Convergence
    for color_idx, idx in enumerate(top_indices[:5]):
        feat_name = features[idx]
        magnitudes = [subset_mean_abs_cand[k][idx] for k in TREE_SUBSETS]
        ax_c.plot(TREE_SUBSETS, magnitudes, marker="o", linewidth=2, label=feat_name)
    ax_c.set_title("(C) Top-5 Feature Attribution Magnitudes Convergence", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Ensemble Size (Trees)", fontsize=10)
    ax_c.set_ylabel("Mean |SHAP Anomaly Attribution|", fontsize=10)
    ax_c.set_xticks(TREE_SUBSETS)
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(loc="upper right", fontsize=8)

    # Panel D: Additivity Maximum Error across K
    ax_d.plot(conv_df["tree_count"], conv_df["max_additivity_error"] * 1e14, marker="D", color="#d62728", linewidth=2)
    ax_d.set_title("(D) TreeSHAP Exact Additivity Error (x 1e-14)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Ensemble Size (Trees)", fontsize=10)
    ax_d.set_ylabel("Max Error vs. Tree Path Length ($10^{-14}$)", fontsize=10)
    ax_d.set_xticks(TREE_SUBSETS)
    ax_d.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig2_path = exp_dir / "ensemble_convergence_metrics.png"
    plt.savefig(fig2_path)
    plt.close(fig2)
    logger.info(f"Saved Figure 2: {fig2_path.name}")

    logger.info("\n" + "=" * 76)
    logger.info("EXP-PHASE3-SHAP-03 STABILITY EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("=" * 76)

    return conv_df, feat_df, sample_df


if __name__ == "__main__":
    run_shap_stability_experiment()

