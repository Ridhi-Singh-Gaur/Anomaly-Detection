"""
Phase 3: Experimental Validation of Context-Aware XAI
Experiment 4: Randomized TreeSHAP Stability Validation
Experiment ID: EXP-PHASE3-SHAP-04

Evaluates whether TreeSHAP explanation stability depends on the deterministic
prefix ordering of the original Isolation Forest trees by drawing 20 independent,
deterministic random subsets without replacement for each tree count:
  - K = 25  (20 random draws)
  - K = 50  (20 random draws)
  - K = 100 (20 random draws)
  - K = 150 (20 random draws)
  Total: 80 randomized subset ensemble runs.

Reference:
  - Full fitted 200-tree Isolation Forest model (Table IX baseline).

Evaluation sample:
  - 333 candidate anomalies (multi-model consensus pool)
  - 100 randomly sampled normal test instances (random_state=42)
  - Total evaluation pool: 433 instances
"""

from pathlib import Path
import sys
import os
import copy
import logging
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, t as t_dist
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

TREE_SIZES = [25, 50, 100, 150]
NUM_TRIALS = 20
BASE_SEED = 42000

# Prefix baseline results from Experiment 3 for direct comparison
EXP3_PREFIX_RESULTS = {
    25: {"spearman": 0.870588, "cosine": 0.956041, "top5_overlap": 80.0, "mean_sample_cos": 0.9086},
    50: {"spearman": 0.917647, "cosine": 0.973453, "top5_overlap": 80.0, "mean_sample_cos": 0.9480},
    100: {"spearman": 0.988235, "cosine": 0.994726, "top5_overlap": 100.0, "mean_sample_cos": 0.9856},
    150: {"spearman": 0.997059, "cosine": 0.998334, "top5_overlap": 100.0, "mean_sample_cos": 0.9961},
}


def calc_ci95(data):
    """Calculate mean, std, and 95% confidence interval using Student's t distribution."""
    n = len(data)
    mean_val = float(np.mean(data))
    std_val = float(np.std(data, ddof=1)) if n > 1 else 0.0
    if n > 1 and std_val > 0:
        se = std_val / np.sqrt(n)
        h = se * t_dist.ppf(0.975, df=n - 1)
        ci_lower = mean_val - h
        ci_upper = mean_val + h
    else:
        ci_lower = mean_val
        ci_upper = mean_val
    return mean_val, std_val, float(ci_lower), float(ci_upper)


def run_randomized_stability_experiment():
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
    logger = logging.getLogger("EXP-PHASE3-SHAP-04")

    logger.info("=" * 76)
    logger.info("MEMBER 4 - PHASE 3: RANDOMIZED TREESHAP STABILITY VALIDATION")
    logger.info("Experiment ID: EXP-PHASE3-SHAP-04")
    logger.info("=" * 76)

    # 1. Load Model and Features
    model_path = src_dir / "outputs" / "isolation_forest_model.joblib"
    saved = joblib.load(model_path)
    model = saved["model"]
    scaler = saved["scaler"]
    features = list(saved["feature_columns"])
    medians = saved["training_medians"]
    total_trees = len(model.estimators_)
    assert total_trees == 200, f"Expected 200 trees, got {total_trees}"
    assert len(model.estimators_features_) == 200
    logger.info(f"Loaded reference Isolation Forest model: {total_trees} trees, {len(features)} features")

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
    logger.info(f"Evaluation sample: {len(combined_eval)} instances (333 candidates + 100 normal)")

    # 3. Preprocess Features
    x_raw = combined_eval[features].replace([np.inf, -np.inf], np.nan).fillna(medians)
    x_scaled = scaler.transform(x_raw)
    x_cand_scaled = x_scaled[:333]

    # 4. Compute Reference 200-Tree SHAP Baseline
    logger.info("\nComputing reference 200-tree TreeSHAP baseline...")
    ref_explainer = shap.TreeExplainer(model)
    ref_shap_all = -ref_explainer.shap_values(x_scaled)  # Inverted anomaly attribution
    ref_shap_cand = ref_shap_all[:333]
    ref_mean_abs = np.mean(np.abs(ref_shap_cand), axis=0)
    ref_order = np.argsort(-ref_mean_abs)
    ref_ranks = np.empty_like(ref_order)
    ref_ranks[ref_order] = np.arange(1, len(features) + 1)
    ref_top5 = [features[i] for i in ref_order[:5]]
    ref_top5_set = set(ref_top5)
    logger.info(f"Reference Top 5: {ref_top5}")

    # Containers for results
    run_records = []
    feature_ranks_by_size = {k: {f: [] for f in features} for k in TREE_SIZES}
    feature_attributions_by_size = {k: {f: [] for f in features} for k in TREE_SIZES}
    feature_top5_counts_by_size = {k: {f: 0 for f in features} for k in TREE_SIZES}

    # 5. Execute 20 Randomized Runs for Each Tree Count
    total_runs = len(TREE_SIZES) * NUM_TRIALS
    run_idx = 0

    for k in TREE_SIZES:
        logger.info(f"\n========================================================")
        logger.info(f"Evaluating K = {k} trees across {NUM_TRIALS} randomized subset trials")
        logger.info(f"========================================================")

        for trial in range(NUM_TRIALS):
            run_idx += 1
            seed = BASE_SEED + k * 100 + trial
            rng = np.random.RandomState(seed)

            # Draw k unique indices without replacement
            selected_indices = np.sort(rng.choice(total_trees, size=k, replace=False)).tolist()
            assert len(selected_indices) == k
            assert len(set(selected_indices)) == k, "Duplicate trees detected in random draw!"

            # Construct valid subset model preserving estimators_features_ alignment
            sub_model = copy.deepcopy(model)
            sub_model.estimators_ = [model.estimators_[i] for i in selected_indices]
            sub_model.estimators_features_ = [model.estimators_features_[i] for i in selected_indices]
            sub_model.n_estimators = k

            # Explain with TreeExplainer
            explainer = shap.TreeExplainer(sub_model)
            exp_val = float(explainer.expected_value[0])
            shap_vals = explainer.shap_values(x_scaled)
            shap_anom = -shap_vals  # Inverted anomaly attributions

            # Additivity check against subset direct path length
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
            max_add_err = float(np.max(add_errs))
            mean_add_err = float(np.mean(add_errs))
            assert max_add_err < 1e-12, f"Additivity error {max_add_err} exceeded tolerance on run {run_idx}!"

            # Global metrics on 333 candidate anomalies
            shap_anom_cand = shap_anom[:333]
            mean_abs_cand = np.mean(np.abs(shap_anom_cand), axis=0)
            order = np.argsort(-mean_abs_cand)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(features) + 1)
            trial_top5 = [features[i] for i in order[:5]]
            trial_top5_set = set(trial_top5)

            # Record feature frequencies
            for f_idx, feat in enumerate(features):
                r_val = int(ranks[f_idx])
                attr_val = float(mean_abs_cand[f_idx])
                feature_ranks_by_size[k][feat].append(r_val)
                feature_attributions_by_size[k][feat].append(attr_val)
                if feat in trial_top5_set:
                    feature_top5_counts_by_size[k][feat] += 1

            # Global alignment vs 200-tree reference
            rho, _ = spearmanr(ranks, ref_ranks)
            cos_global = float(cosine_similarity(mean_abs_cand.reshape(1, -1), ref_mean_abs.reshape(1, -1))[0, 0])
            top5_overlap = len(trial_top5_set.intersection(ref_top5_set))
            top5_overlap_pct = (top5_overlap / 5.0) * 100.0
            top5_jaccard = top5_overlap / len(trial_top5_set.union(ref_top5_set))
            exact_top5_recovered = bool(top5_overlap == 5)
            exact_top5_ordered_recovered = bool(trial_top5 == ref_top5)

            # Instance-level similarity across all 433 instances
            sample_cos_list = []
            sample_mae_list = []
            sample_l2_list = []
            for i in range(len(combined_eval)):
                v_trial = shap_anom[i].reshape(1, -1)
                v_ref = ref_shap_all[i].reshape(1, -1)
                sample_cos_list.append(float(cosine_similarity(v_trial, v_ref)[0, 0]))
                sample_mae_list.append(float(np.mean(np.abs(v_trial - v_ref))))
                sample_l2_list.append(float(np.linalg.norm(v_trial - v_ref)))

            mean_s_cos = float(np.mean(sample_cos_list))
            med_s_cos = float(np.median(sample_cos_list))
            cand_mean_cos = float(np.mean(sample_cos_list[:333]))
            norm_mean_cos = float(np.mean(sample_cos_list[333:]))
            mean_s_mae = float(np.mean(sample_mae_list))
            mean_s_l2 = float(np.mean(sample_l2_list))

            rec = {
                "run_id": run_idx,
                "tree_count": k,
                "trial": trial + 1,
                "seed": seed,
                "expected_path_length": exp_val,
                "max_additivity_error": max_add_err,
                "mean_additivity_error": mean_add_err,
                "spearman_rho": float(rho),
                "cosine_similarity": cos_global,
                "top5_overlap_count": top5_overlap,
                "top5_overlap_pct": top5_overlap_pct,
                "top5_jaccard": float(top5_jaccard),
                "exact_top5_recovered": exact_top5_recovered,
                "exact_top5_ordered_recovered": exact_top5_ordered_recovered,
                "top5_features": ";".join(trial_top5),
                "mean_sample_cosine": mean_s_cos,
                "median_sample_cosine": med_s_cos,
                "candidate_mean_cosine": cand_mean_cos,
                "normal_mean_cosine": norm_mean_cos,
                "mean_sample_mae": mean_s_mae,
                "mean_sample_l2": mean_s_l2,
                "selected_indices_str": ",".join(map(str, selected_indices)),
            }
            run_records.append(rec)

            if (trial + 1) % 5 == 0 or trial == 0:
                logger.info(f"  [K={k:3d} Trial {trial+1:02d}/20] Rho={rho:.4f} | Cosine={cos_global:.4f} | Top5 Overlap={top5_overlap_pct:.0f}% | Sample Cosine={mean_s_cos:.4f}")

    # 6. Export Per-Run Summary CSV
    run_df = pd.DataFrame(run_records)
    run_csv = exp_dir / "randomized_runs_summary.csv"
    run_df.to_csv(run_csv, index=False)
    logger.info(f"\nSaved randomized runs summary CSV ({len(run_df)} runs): {run_csv.name}")

    # 7. Aggregate Statistics and 95% Confidence Intervals per Tree Count
    logger.info("\nAggregating statistical convergence metrics across randomized trials...")
    summary_records = []

    for k in TREE_SIZES:
        k_df = run_df[run_df["tree_count"] == k]
        prefix_res = EXP3_PREFIX_RESULTS[k]

        rho_mean, rho_std, rho_ci_l, rho_ci_u = calc_ci95(k_df["spearman_rho"])
        cos_mean, cos_std, cos_ci_l, cos_ci_u = calc_ci95(k_df["cosine_similarity"])
        s_cos_mean, s_cos_std, s_cos_ci_l, s_cos_ci_u = calc_ci95(k_df["mean_sample_cosine"])
        mae_mean, mae_std, mae_ci_l, mae_ci_u = calc_ci95(k_df["mean_sample_mae"])

        top5_ov_mean = float(k_df["top5_overlap_pct"].mean())
        top5_ov_std = float(k_df["top5_overlap_pct"].std())
        exact_top5_rate = float(k_df["exact_top5_recovered"].mean() * 100.0)
        exact_ordered_rate = float(k_df["exact_top5_ordered_recovered"].mean() * 100.0)

        # Compare prefix baseline against distribution
        prefix_rho = prefix_res["spearman"]
        prefix_cos = prefix_res["cosine"]
        prefix_in_range_rho = bool(k_df["spearman_rho"].min() <= prefix_rho <= k_df["spearman_rho"].max())
        prefix_in_range_cos = bool(k_df["cosine_similarity"].min() <= prefix_cos <= k_df["cosine_similarity"].max())

        row = {
            "tree_count": k,
            "num_trials": NUM_TRIALS,
            # Spearman Rho
            "spearman_mean": rho_mean,
            "spearman_std": rho_std,
            "spearman_min": float(k_df["spearman_rho"].min()),
            "spearman_max": float(k_df["spearman_rho"].max()),
            "spearman_ci95_lower": rho_ci_l,
            "spearman_ci95_upper": rho_ci_u,
            # Cosine Similarity
            "cosine_mean": cos_mean,
            "cosine_std": cos_std,
            "cosine_min": float(k_df["cosine_similarity"].min()),
            "cosine_max": float(k_df["cosine_similarity"].max()),
            "cosine_ci95_lower": cos_ci_l,
            "cosine_ci95_upper": cos_ci_u,
            # Top 5 Overlap
            "top5_overlap_mean_pct": top5_ov_mean,
            "top5_overlap_std_pct": top5_ov_std,
            "top5_overlap_min_pct": float(k_df["top5_overlap_pct"].min()),
            "top5_overlap_max_pct": float(k_df["top5_overlap_pct"].max()),
            "exact_top5_recovery_rate_pct": exact_top5_rate,
            "exact_top5_ordered_rate_pct": exact_ordered_rate,
            # Per-Sample Cosine
            "sample_cosine_mean": s_cos_mean,
            "sample_cosine_std": s_cos_std,
            "sample_cosine_min": float(k_df["mean_sample_cosine"].min()),
            "sample_cosine_max": float(k_df["mean_sample_cosine"].max()),
            "sample_cosine_ci95_lower": s_cos_ci_l,
            "sample_cosine_ci95_upper": s_cos_ci_u,
            # Per-Sample MAE
            "sample_mae_mean": mae_mean,
            "sample_mae_std": mae_std,
            "sample_mae_min": float(k_df["mean_sample_mae"].min()),
            "sample_mae_max": float(k_df["mean_sample_mae"].max()),
            # Comparison with Exp 3 Prefix Baseline
            "exp3_prefix_spearman": prefix_rho,
            "exp3_prefix_cosine": prefix_cos,
            "prefix_rho_within_range": prefix_in_range_rho,
            "prefix_cos_within_range": prefix_in_range_cos,
        }
        summary_records.append(row)

        logger.info(f"\n[SUMMARY K={k:3d}]")
        logger.info(f"  Spearman Rho: Mean = {rho_mean:.4f} +/- {rho_std:.4f} (95% CI: [{rho_ci_l:.4f}, {rho_ci_u:.4f}], Range: [{row['spearman_min']:.4f}, {row['spearman_max']:.4f}])")
        logger.info(f"  Cosine Sim:   Mean = {cos_mean:.6f} +/- {cos_std:.6f} (95% CI: [{cos_ci_l:.6f}, {cos_ci_u:.6f}], Range: [{row['cosine_min']:.6f}, {row['cosine_max']:.6f}])")
        logger.info(f"  Top-5 Overlap: Mean = {top5_ov_mean:.1f}% | Exact Set Recovery Rate = {exact_top5_rate:.1f}% | Exact Ordered = {exact_ordered_rate:.1f}%")
        logger.info(f"  Sample Cosine: Mean = {s_cos_mean:.4f} +/- {s_cos_std:.4f} (95% CI: [{s_cos_ci_l:.4f}, {s_cos_ci_u:.4f}])")
        logger.info(f"  Exp 3 Prefix Fall within Random Range: Rho={prefix_in_range_rho}, Cosine={prefix_in_range_cos}")

    summary_df = pd.DataFrame(summary_records)
    summary_csv = exp_dir / "randomized_stability_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved randomized stability summary CSV: {summary_csv.name}")

    # 8. Feature Selection Frequency and Ranking Stability CSV
    logger.info("\nComputing feature selection frequencies and rank stability...")
    freq_records = []
    for f_idx, feat in enumerate(features):
        row = {
            "feature": feat,
            "ref_rank_200": int(ref_ranks[f_idx]),
            "ref_mean_abs_shap_200": float(ref_mean_abs[f_idx]),
        }
        for k in TREE_SIZES:
            ranks = feature_ranks_by_size[k][feat]
            attrs = feature_attributions_by_size[k][feat]
            top5_cnt = feature_top5_counts_by_size[k][feat]
            top5_pct = (top5_cnt / NUM_TRIALS) * 100.0

            row[f"top5_count_k{k}"] = top5_cnt
            row[f"top5_pct_k{k}"] = top5_pct
            row[f"rank_mean_k{k}"] = float(np.mean(ranks))
            row[f"rank_std_k{k}"] = float(np.std(ranks))
            row[f"rank_min_k{k}"] = int(np.min(ranks))
            row[f"rank_max_k{k}"] = int(np.max(ranks))
            row[f"attr_mean_k{k}"] = float(np.mean(attrs))
            row[f"attr_std_k{k}"] = float(np.std(attrs))

        freq_records.append(row)

    freq_df = pd.DataFrame(freq_records).sort_values("ref_rank_200").reset_index(drop=True)
    freq_csv = exp_dir / "feature_top5_frequency_stability.csv"
    freq_df.to_csv(freq_csv, index=False)
    logger.info(f"Saved feature top-5 frequency stability CSV: {freq_csv.name}")

    # 9. Publication-Quality Visualizations
    logger.info("\nGenerating publication figures...")

    # Figure 1: Randomized Ensemble Convergence (Multi-panel with 95% CIs and Prefix Overlay)
    fig1, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # Panel A: Spearman Rank Correlation Distribution Across Randomized Draws
    box_rhos = [run_df[run_df["tree_count"] == k]["spearman_rho"] for k in TREE_SIZES]
    bp_a = ax_a.boxplot(box_rhos, patch_artist=True, positions=TREE_SIZES, widths=8)
    for patch in bp_a["boxes"]:
        patch.set_facecolor("#cfe2f3")
        patch.set_edgecolor("#1f77b4")
    # Overlay individual runs as jitter
    for k in TREE_SIZES:
        rhos_k = run_df[run_df["tree_count"] == k]["spearman_rho"]
        jitter = np.random.RandomState(42).uniform(-2, 2, size=len(rhos_k))
        ax_a.scatter(k + jitter, rhos_k, color="#1f77b4", alpha=0.6, s=25, zorder=3)
        # Prefix baseline marker
        ax_a.scatter(k, EXP3_PREFIX_RESULTS[k]["spearman"], marker="D", color="#d62728", s=65, zorder=4, label="Exp 3 Prefix Baseline" if k == 25 else "")
    # Connect means with 95% CI error bars
    means_rho = summary_df["spearman_mean"]
    errors_rho = [summary_df["spearman_mean"] - summary_df["spearman_ci95_lower"],
                  summary_df["spearman_ci95_upper"] - summary_df["spearman_mean"]]
    ax_a.errorbar(TREE_SIZES, means_rho, yerr=errors_rho, fmt="o-", color="#08519c", linewidth=2, capsize=5, label="Randomized Mean (95% CI)", zorder=5)
    ax_a.set_title("(A) Spearman Rank Correlation vs. 200-Tree Baseline", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Ensemble Tree Count (K)", fontsize=10)
    ax_a.set_ylabel("Spearman Rank Correlation (ρ)", fontsize=10)
    ax_a.set_xticks(TREE_SIZES)
    ax_a.set_ylim(0.70, 1.02)
    ax_a.grid(True, linestyle=":", alpha=0.5)
    ax_a.legend(loc="lower right", fontsize=9)

    # Panel B: Global Cosine Similarity Distribution
    box_cos = [run_df[run_df["tree_count"] == k]["cosine_similarity"] for k in TREE_SIZES]
    bp_b = ax_b.boxplot(box_cos, patch_artist=True, positions=TREE_SIZES, widths=8)
    for patch in bp_b["boxes"]:
        patch.set_facecolor("#d9f0d3")
        patch.set_edgecolor("#2ca02c")
    for k in TREE_SIZES:
        cos_k = run_df[run_df["tree_count"] == k]["cosine_similarity"]
        jitter = np.random.RandomState(42).uniform(-2, 2, size=len(cos_k))
        ax_b.scatter(k + jitter, cos_k, color="#2ca02c", alpha=0.6, s=25, zorder=3)
        ax_b.scatter(k, EXP3_PREFIX_RESULTS[k]["cosine"], marker="D", color="#d62728", s=65, zorder=4, label="Exp 3 Prefix Baseline" if k == 25 else "")
    means_cos = summary_df["cosine_mean"]
    errors_cos = [summary_df["cosine_mean"] - summary_df["cosine_ci95_lower"],
                  summary_df["cosine_ci95_upper"] - summary_df["cosine_mean"]]
    ax_b.errorbar(TREE_SIZES, means_cos, yerr=errors_cos, fmt="s-", color="#006d2c", linewidth=2, capsize=5, label="Randomized Mean (95% CI)", zorder=5)
    ax_b.set_title("(B) Global Attribution Vector Cosine Similarity", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Ensemble Tree Count (K)", fontsize=10)
    ax_b.set_ylabel("Cosine Similarity", fontsize=10)
    ax_b.set_xticks(TREE_SIZES)
    ax_b.set_ylim(0.90, 1.005)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    ax_b.legend(loc="lower right", fontsize=9)

    # Panel C: Per-Sample Mean Cosine Similarity (Candidates vs Normal)
    cand_means = [run_df[run_df["tree_count"] == k]["candidate_mean_cosine"].mean() for k in TREE_SIZES]
    norm_means = [run_df[run_df["tree_count"] == k]["normal_mean_cosine"].mean() for k in TREE_SIZES]
    ax_c.plot(TREE_SIZES, cand_means, marker="o", linewidth=2.2, color="#e41a1c", label="Candidate Anomalies (N=333)")
    ax_c.plot(TREE_SIZES, norm_means, marker="s", linewidth=2.2, color="#377eb8", label="Normal Observations (N=100)")
    ax_c.set_title("(C) Per-Sample Attribution Similarity to 200-Tree Baseline", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Ensemble Tree Count (K)", fontsize=10)
    ax_c.set_ylabel("Mean Instance Cosine Similarity", fontsize=10)
    ax_c.set_xticks(TREE_SIZES)
    ax_c.set_ylim(0.80, 1.01)
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(loc="lower right", fontsize=9)

    # Panel D: Exact Top-5 Recovery Rate (%)
    rec_rates = summary_df["exact_top5_recovery_rate_pct"]
    ordered_rates = summary_df["exact_top5_ordered_rate_pct"]
    bars_w = 6
    ax_d.bar([k - 3 for k in TREE_SIZES], rec_rates, width=bars_w, color="#756bb1", label="Exact Top-5 Set Recovery (100% Overlap)")
    ax_d.bar([k + 3 for k in TREE_SIZES], ordered_rates, width=bars_w, color="#bcbddc", label="Exact Ordered Top-5 Recovery")
    for k, r, o in zip(TREE_SIZES, rec_rates, ordered_rates):
        ax_d.text(k - 3, r + 2, f"{r:.0f}%", ha="center", fontsize=9, fontweight="bold")
        ax_d.text(k + 3, o + 2, f"{o:.0f}%", ha="center", fontsize=8)
    ax_d.set_title("(D) Top-Five Feature Set Recovery Rate across 20 Draws", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Ensemble Tree Count (K)", fontsize=10)
    ax_d.set_ylabel("Recovery Rate across Trials (%)", fontsize=10)
    ax_d.set_xticks(TREE_SIZES)
    ax_d.set_ylim(0, 115)
    ax_d.grid(True, linestyle=":", alpha=0.5)
    ax_d.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig1_path = exp_dir / "randomized_ensemble_convergence.png"
    plt.savefig(fig1_path)
    plt.close(fig1)
    logger.info(f"Saved Figure 1: {fig1_path.name}")

    # Figure 2: Top-5 Feature Selection Frequency Across Configurations
    fig2, ax2 = plt.subplots(figsize=(12, 6), dpi=300)
    top_indices = np.argsort(ref_ranks)[:8]  # Top 8 features
    feat_names = [features[i] for i in top_indices]
    n_feats = len(feat_names)
    bar_width = 0.18
    x_positions = np.arange(n_feats)

    palette = ["#fee5d9", "#fcae91", "#fb6a4a", "#cb181d"]
    for idx_k, k in enumerate(TREE_SIZES):
        pcts = [feature_top5_counts_by_size[k][f] * 5.0 for f in feat_names]  # Count out of 20 -> %
        ax2.bar(x_positions + idx_k * bar_width, pcts, width=bar_width, color=palette[idx_k], label=f"K = {k} trees")

    ax2.set_title("Top-Five Selection Frequency (%) Across 20 Randomized Ensemble Draws", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Feature Name (Ordered by 200-Tree Reference Rank)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Selection Frequency in Top-5 (%)", fontsize=11, fontweight="semibold")
    ax2.set_xticks(x_positions + bar_width * 1.5)
    ax2.set_xticklabels(feat_names, rotation=25, ha="right", fontsize=10, fontweight="semibold")
    ax2.set_ylim(0, 115)
    ax2.axhline(100.0, color="#666666", linestyle="--", alpha=0.7, label="100% Invariant Selection Threshold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    fig2_path = exp_dir / "feature_top5_selection_frequency.png"
    plt.savefig(fig2_path)
    plt.close(fig2)
    logger.info(f"Saved Figure 2: {fig2_path.name}")

    logger.info("\n" + "=" * 76)
    logger.info("EXP-PHASE3-SHAP-04 EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("=" * 76)

    return summary_df, run_df, freq_df


if __name__ == "__main__":
    run_randomized_stability_experiment()

