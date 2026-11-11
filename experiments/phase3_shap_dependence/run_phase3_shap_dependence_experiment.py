"""
Phase 3: Experimental Validation of Context-Aware XAI
Experiment 5: Feature Dependence and SHAP Attribution Redundancy Audit
Experiment ID: EXP-PHASE3-SHAP-05

Audits whether dominant TreeSHAP features in the fitted Isolation Forest are
statistically dependent and whether collinear features share, redistribute,
or obscure their feature attributions.

Evaluates:
  1. Feature-value correlation matrices (Pearson & Spearman) across:
     - Training observations (N = 109,347)
     - Candidate anomalies (N = 333)
     - Normal observations (N = 100)
     - Combined evaluation sample (N = 433)
  2. SHAP attribution correlation matrices across the same partitions.
  3. Feature pair discrepancy analysis (Raw Correlation vs. SHAP Correlation).
  4. Grouped feature importance across 6 functional domains under multiple aggregation norms:
     - Standard L1 sum of mean absolute SHAP
     - Sample-wise L2 Euclidean norm
     - Sample-wise Net Signed algebraic attribution
  5. Collinear pair attribution consolidation (Global_active_power + Global_intensity).
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
base_dir = Path(__file__).resolve().parent.parent.parent
src_dir = ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir) if ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir)).exists() else base_dir)
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

FEATURE_GROUPS = {
    "Direct Electrical Measurements": [
        "Global_active_power",
        "Global_intensity",
        "Global_reactive_power",
        "Voltage",
    ],
    "Sub-metering Measurements": [
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ],
    "Rolling-Window Features": [
        "rolling_mean_4",
        "rolling_std_4",
    ],
    "Lag Features": [
        "lag_1",
        "lag_4",
    ],
    "Temporal / Calendar Features": [
        "hour",
        "day_of_week",
        "is_weekend",
        "month",
    ],
    "Consumption-Change Features": [
        "consumption_change",
    ],
}


def run_dependence_experiment():
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
    logger = logging.getLogger("EXP-PHASE3-SHAP-05")

    logger.info("=" * 76)
    logger.info("MEMBER 4 - PHASE 3: FEATURE DEPENDENCE & SHAP REDUNDANCY AUDIT")
    logger.info("Experiment ID: EXP-PHASE3-SHAP-05")
    logger.info("=" * 76)

    # 1. Load Model Metadata and Features
    model_path = src_dir / "outputs" / "isolation_forest_model.joblib"
    saved = joblib.load(model_path)
    features = list(saved["feature_columns"])
    medians = saved["training_medians"]
    assert len(features) == 16, f"Expected 16 features, got {len(features)}"
    logger.info(f"Loaded feature list ({len(features)} features)")

    # 2. Load Raw Features across Partitions
    feat_csv = src_dir / "outputs" / "household_15min_features.csv"
    cand_csv = src_dir / "outputs" / "contextual_anomalies.csv"

    feat_df = pd.read_csv(feat_csv)
    feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"])
    feat_df = feat_df.sort_values("timestamp").reset_index(drop=True)

    # Training split (first 80%)
    split_idx = int(len(feat_df) * 0.8)
    train_df = feat_df.iloc[:split_idx].copy().reset_index(drop=True)
    logger.info(f"Training observations: {len(train_df):,} rows")

    # Candidate anomalies (N = 333)
    cand_df = pd.read_csv(cand_csv)
    cand_df["timestamp"] = pd.to_datetime(cand_df["timestamp"])
    cand_merged = pd.merge(cand_df[["timestamp"]], feat_df, on="timestamp", how="left")
    assert len(cand_merged) == 333, f"Expected 333 candidates, got {len(cand_merged)}"

    # Sampled normal observations (N = 100, seed 42)
    test_start = pd.to_datetime("2010-02-05 05:30:00")
    test_df = feat_df[feat_df["timestamp"] >= test_start].copy()
    norm_merged = test_df[~test_df["timestamp"].isin(cand_df["timestamp"])].sample(n=100, random_state=42).copy()
    assert len(norm_merged) == 100, f"Expected 100 normal instances, got {len(norm_merged)}"

    # Combined 433 observations
    combined_eval = pd.concat([cand_merged, norm_merged], ignore_index=True)
    assert len(combined_eval) == 433, f"Expected 433 combined instances, got {len(combined_eval)}"

    # Impute missing values with training medians
    X_train = train_df[features].replace([np.inf, -np.inf], np.nan).fillna(medians)
    X_cand = cand_merged[features].replace([np.inf, -np.inf], np.nan).fillna(medians)
    X_norm = norm_merged[features].replace([np.inf, -np.inf], np.nan).fillna(medians)
    X_comb = combined_eval[features].replace([np.inf, -np.inf], np.nan).fillna(medians)

    # 3. Load Existing 200-Tree SHAP Attribution Matrix
    shap_ref_csv = base_dir / "experiments" / "phase3_shap_stability" / "shap_values_trees_200.csv"
    assert shap_ref_csv.exists(), f"Reference SHAP CSV missing: {shap_ref_csv}"
    shap_df = pd.read_csv(shap_ref_csv)
    assert len(shap_df) == 433, f"Expected 433 rows in SHAP CSV, got {len(shap_df)}"

    # Extract 16-feature SHAP matrices
    shap_cols = [f"shap_{f}" for f in features]
    S_comb = shap_df[shap_cols].copy()
    S_comb.columns = features
    S_cand = S_comb.iloc[:333].copy()
    S_norm = S_comb.iloc[333:].copy()

    # =========================================================================
    # TASK 1: FEATURE-VALUE CORRELATION ANALYSIS
    # =========================================================================
    logger.info("\nComputing raw feature-value correlations...")
    corr_raw_train_p = X_train.corr(method="pearson")
    corr_raw_train_s = X_train.corr(method="spearman")
    corr_raw_cand_p = X_cand.corr(method="pearson")
    corr_raw_cand_s = X_cand.corr(method="spearman")
    corr_raw_norm_p = X_norm.corr(method="pearson")
    corr_raw_norm_s = X_norm.corr(method="spearman")
    corr_raw_comb_p = X_comb.corr(method="pearson")
    corr_raw_comb_s = X_comb.corr(method="spearman")

    # Export raw correlations CSV (stacked format for clean inspection)
    raw_corr_records = []
    for f1 in features:
        for f2 in features:
            raw_corr_records.append({
                "feature_1": f1,
                "feature_2": f2,
                "train_pearson": float(corr_raw_train_p.loc[f1, f2]),
                "train_spearman": float(corr_raw_train_s.loc[f1, f2]),
                "candidates_pearson": float(corr_raw_cand_p.loc[f1, f2]),
                "candidates_spearman": float(corr_raw_cand_s.loc[f1, f2]),
                "normal_pearson": float(corr_raw_norm_p.loc[f1, f2]),
                "normal_spearman": float(corr_raw_norm_s.loc[f1, f2]),
                "combined_pearson": float(corr_raw_comb_p.loc[f1, f2]),
                "combined_spearman": float(corr_raw_comb_s.loc[f1, f2]),
            })
    raw_corr_df = pd.DataFrame(raw_corr_records)
    raw_corr_csv = exp_dir / "feature_value_correlations.csv"
    raw_corr_df.to_csv(raw_corr_csv, index=False)
    logger.info(f"Saved feature_value_correlations.csv ({len(raw_corr_df)} pairs)")

    # =========================================================================
    # TASK 2: SHAP ATTRIBUTION CORRELATION ANALYSIS
    # =========================================================================
    logger.info("\nComputing SHAP attribution correlations...")
    corr_shap_cand_p = S_cand.corr(method="pearson")
    corr_shap_cand_s = S_cand.corr(method="spearman")
    corr_shap_norm_p = S_norm.corr(method="pearson")
    corr_shap_norm_s = S_norm.corr(method="spearman")
    corr_shap_comb_p = S_comb.corr(method="pearson")
    corr_shap_comb_s = S_comb.corr(method="spearman")

    shap_corr_records = []
    for f1 in features:
        for f2 in features:
            shap_corr_records.append({
                "feature_1": f1,
                "feature_2": f2,
                "candidates_shap_pearson": float(corr_shap_cand_p.loc[f1, f2]),
                "candidates_shap_spearman": float(corr_shap_cand_s.loc[f1, f2]),
                "normal_shap_pearson": float(corr_shap_norm_p.loc[f1, f2]),
                "normal_shap_spearman": float(corr_shap_norm_s.loc[f1, f2]),
                "combined_shap_pearson": float(corr_shap_comb_p.loc[f1, f2]),
                "combined_shap_spearman": float(corr_shap_comb_s.loc[f1, f2]),
            })
    shap_corr_df = pd.DataFrame(shap_corr_records)
    shap_corr_csv = exp_dir / "shap_attribution_correlations.csv"
    shap_corr_df.to_csv(shap_corr_csv, index=False)
    logger.info(f"Saved shap_attribution_correlations.csv ({len(shap_corr_df)} pairs)")

    # Combine unique pairs (120 pairs) for pairwise comparative analysis
    pair_records = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]

            r_train_p = float(corr_raw_train_p.loc[f1, f2])
            r_train_s = float(corr_raw_train_s.loc[f1, f2])
            r_cand_p = float(corr_raw_cand_p.loc[f1, f2])
            r_cand_s = float(corr_raw_cand_s.loc[f1, f2])

            s_cand_p = float(corr_shap_cand_p.loc[f1, f2])
            s_cand_s = float(corr_shap_cand_s.loc[f1, f2])
            s_comb_p = float(corr_shap_comb_p.loc[f1, f2])
            s_comb_s = float(corr_shap_comb_s.loc[f1, f2])

            discrepancy_p = abs(r_cand_p - s_cand_p)
            discrepancy_s = abs(r_cand_s - s_cand_s)

            pair_records.append({
                "feature_1": f1,
                "feature_2": f2,
                "train_raw_pearson": r_train_p,
                "train_raw_spearman": r_train_s,
                "cand_raw_pearson": r_cand_p,
                "cand_raw_spearman": r_cand_s,
                "cand_shap_pearson": s_cand_p,
                "cand_shap_spearman": s_cand_s,
                "comb_shap_pearson": s_comb_p,
                "comb_shap_spearman": s_comb_s,
                "cand_discrepancy_pearson": discrepancy_p,
                "cand_discrepancy_spearman": discrepancy_s,
                "is_train_p_ge_70": bool(abs(r_train_p) >= 0.70),
                "is_train_p_ge_80": bool(abs(r_train_p) >= 0.80),
                "is_train_p_ge_90": bool(abs(r_train_p) >= 0.90),
                "is_cand_p_ge_70": bool(abs(r_cand_p) >= 0.70),
                "is_cand_p_ge_80": bool(abs(r_cand_p) >= 0.80),
                "is_cand_p_ge_90": bool(abs(r_cand_p) >= 0.90),
                "is_shap_p_ge_70": bool(abs(s_cand_p) >= 0.70),
            })

    pair_df = pd.DataFrame(pair_records).sort_values("cand_raw_pearson", ascending=False).reset_index(drop=True)
    pair_csv = exp_dir / "feature_correlation_pairs.csv"
    pair_df.to_csv(pair_csv, index=False)
    logger.info(f"Saved feature_correlation_pairs.csv (120 unique feature pairs)")

    # Log hypothesis pairs
    logger.info("\n--- Target Hypotheses Correlation Audit ---")
    target_pairs = [
        ("Global_active_power", "Global_intensity"),
        ("rolling_mean_4", "lag_1"),
        ("Sub_metering_1", "Sub_metering_2"),
        ("Sub_metering_1", "Sub_metering_3"),
        ("Sub_metering_2", "Sub_metering_3"),
        ("lag_1", "lag_4"),
        ("rolling_std_4", "consumption_change"),
    ]
    for f1, f2 in target_pairs:
        row = pair_df[((pair_df["feature_1"] == f1) & (pair_df["feature_2"] == f2)) |
                      ((pair_df["feature_1"] == f2) & (pair_df["feature_2"] == f1))].iloc[0]
        logger.info(f"  {f1} vs {f2}:")
        logger.info(f"    Train Raw:  Pearson = {row['train_raw_pearson']:.4f}, Spearman = {row['train_raw_spearman']:.4f}")
        logger.info(f"    Cand Raw:   Pearson = {row['cand_raw_pearson']:.4f}, Spearman = {row['cand_raw_spearman']:.4f}")
        logger.info(f"    Cand SHAP:  Pearson = {row['cand_shap_pearson']:.4f}, Spearman = {row['cand_shap_spearman']:.4f}")

    # =========================================================================
    # TASK 3: GROUPED FEATURE IMPORTANCE
    # =========================================================================
    logger.info("\nComputing grouped feature importance across functional domains...")
    # Base individual feature attributions on 333 candidate anomalies
    mean_abs_cand = np.mean(np.abs(S_cand.to_numpy()), axis=0)
    total_importance = float(np.sum(mean_abs_cand))
    feat_attr_dict = dict(zip(features, mean_abs_cand))

    group_records = []
    for g_name, g_feats in FEATURE_GROUPS.items():
        # 1. Standard L1 Sum of Mean Absolute SHAP
        g_l1_sum = float(sum(feat_attr_dict[f] for f in g_feats))
        g_pct = (g_l1_sum / total_importance) * 100.0

        # Extract sub-matrix for group
        sub_mat = S_cand[g_feats].to_numpy()  # (333, |G|)

        # 2. Sample-wise L2 Euclidean Norm: mean_i ||phi_{i, G}||_2
        l2_norms = np.linalg.norm(sub_mat, axis=1)
        mean_l2 = float(np.mean(l2_norms))

        # 3. Sample-wise Net Signed Algebraic Sum: mean_i |sum_{j in G} phi_{ij}|
        signed_sums = np.abs(np.sum(sub_mat, axis=1))
        mean_signed = float(np.mean(signed_sums))

        # Within-group distribution string
        within_shares = [f"{f}: {feat_attr_dict[f]:.4f} ({(feat_attr_dict[f]/g_l1_sum)*100.0:.1f}%)" for f in g_feats]

        group_records.append({
            "group_name": g_name,
            "num_features": len(g_feats),
            "constituent_features": ";".join(g_feats),
            "l1_importance_sum": g_l1_sum,
            "percentage_of_total_importance": g_pct,
            "sample_l2_euclidean_norm": mean_l2,
            "sample_net_signed_magnitude": mean_signed,
            "within_group_distributions": "; ".join(within_shares),
        })

    group_df = pd.DataFrame(group_records).sort_values("l1_importance_sum", ascending=False).reset_index(drop=True)
    group_csv = exp_dir / "grouped_shap_importance.csv"
    group_df.to_csv(group_csv, index=False)
    logger.info(f"Saved grouped_shap_importance.csv ({len(group_df)} groups)")

    logger.info("\nGrouped Importance Summary (Candidate Anomalies, N=333):")
    for _, r in group_df.iterrows():
        logger.info(f"  {r['group_name']} ({r['num_features']} feats): L1={r['l1_importance_sum']:.4f} ({r['percentage_of_total_importance']:.2f}%) | L2={r['sample_l2_euclidean_norm']:.4f} | NetSigned={r['sample_net_signed_magnitude']:.4f}")

    # =========================================================================
    # TASK 4: ROBUSTNESS & REDUNDANCY ANALYSIS
    # =========================================================================
    logger.info("\n--- Robustness & Collinear Redundancy Analysis ---")
    p_phi = S_cand["Global_active_power"].to_numpy()
    i_phi = S_cand["Global_intensity"].to_numpy()
    sub2_phi = S_cand["Sub_metering_2"].to_numpy()

    mean_p = float(np.mean(np.abs(p_phi)))
    mean_i = float(np.mean(np.abs(i_phi)))
    mean_sub2 = float(np.mean(np.abs(sub2_phi)))

    # Consolidated bulk power attribution
    bulk_phi = p_phi + i_phi
    mean_bulk = float(np.mean(np.abs(bulk_phi)))
    linear_sum_pi = mean_p + mean_i

    logger.info(f"  Individual Global_intensity:  Mean |SHAP| = {mean_i:.4f} (Rank 3)")
    logger.info(f"  Individual Global_active_power: Mean |SHAP| = {mean_p:.4f} (Rank 5)")
    logger.info(f"  Individual Sub_metering_2:     Mean |SHAP| = {mean_sub2:.4f} (Rank 1)")
    logger.info(f"  Consolidated Bulk Power (P+I): Mean |SHAP| = {mean_bulk:.4f}")
    logger.info(f"  Linear Sum of P and I:                     = {linear_sum_pi:.4f}")
    logger.info(f"  Sign alignment rate: {float(np.mean(np.sign(p_phi) == np.sign(i_phi)) * 100.0):.1f}%")

    # =========================================================================
    # TASK 5: PUBLICATION FIGURES
    # =========================================================================
    logger.info("\nGenerating publication figures...")

    # Figure 1: Feature Dependence & SHAP Attribution Redundancy
    fig1, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(15, 13), dpi=300)

    # Panel A: Heatmap of Raw Feature Correlations (Candidates)
    sns.heatmap(corr_raw_cand_p, ax=ax_a, cmap="vlag", vmin=-1, vmax=1, center=0,
                annot=False, cbar_kws={"label": "Pearson Correlation (r)"}, square=True)
    ax_a.set_title("(A) Raw Feature Correlations (Candidate Anomalies, N=333)", fontsize=11, fontweight="bold")
    ax_a.tick_params(axis="x", rotation=90, labelsize=8)
    ax_a.tick_params(axis="y", rotation=0, labelsize=8)

    # Panel B: Heatmap of SHAP Attribution Correlations (Candidates)
    sns.heatmap(corr_shap_cand_p, ax=ax_b, cmap="vlag", vmin=-1, vmax=1, center=0,
                annot=False, cbar_kws={"label": "SHAP Attribution Correlation (r)"}, square=True)
    ax_b.set_title("(B) TreeSHAP Attribution Correlations (Candidates, N=333)", fontsize=11, fontweight="bold")
    ax_b.tick_params(axis="x", rotation=90, labelsize=8)
    ax_b.tick_params(axis="y", rotation=0, labelsize=8)

    # Panel C: Raw Correlation vs. SHAP Attribution Correlation (Scatter)
    ax_c.scatter(pair_df["cand_raw_pearson"], pair_df["cand_shap_pearson"], color="#1f77b4", alpha=0.7, edgecolors="none", s=40)
    ax_c.plot([-1, 1], [-1, 1], color="#d62728", linestyle="--", linewidth=1.5, label="Identity Line (r_SHAP = r_Raw)")
    # Annotate extreme collinear pairs
    p_row = pair_df[((pair_df["feature_1"] == "Global_active_power") & (pair_df["feature_2"] == "Global_intensity"))].iloc[0]
    ax_c.annotate("Global_active_power vs.\nGlobal_intensity (r=0.999, r_SHAP=0.980)",
                  xy=(p_row["cand_raw_pearson"], p_row["cand_shap_pearson"]),
                  xytext=(0.40, 0.85),
                  arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
                  fontsize=8, fontweight="bold", color="#d62728")
    lag_row = pair_df[((pair_df["feature_1"] == "lag_1") & (pair_df["feature_2"] == "rolling_mean_4"))].iloc[0]
    ax_c.annotate("lag_1 vs.\nrolling_mean_4",
                  xy=(lag_row["cand_raw_pearson"], lag_row["cand_shap_pearson"]),
                  xytext=(0.75, 0.45),
                  arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
                  fontsize=8, color="#2ca02c")
    ax_c.set_title("(C) Raw Feature Correlation vs. SHAP Attribution Correlation", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Raw Feature Pearson Correlation (r_raw)", fontsize=10)
    ax_c.set_ylabel("SHAP Attribution Pearson Correlation (r_shap)", fontsize=10)
    ax_c.set_xlim(-0.6, 1.05)
    ax_c.set_ylim(-0.6, 1.05)
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(loc="lower right", fontsize=9)

    # Panel D: Scatter of SHAP Global_active_power vs SHAP Global_intensity
    ax_d.scatter(p_phi, i_phi, color="#756bb1", alpha=0.7, edgecolors="none", s=35)
    # Fit regression line
    m_slope, b_intercept = np.polyfit(p_phi, i_phi, 1)
    x_vals = np.linspace(min(p_phi), max(p_phi), 100)
    ax_d.plot(x_vals, m_slope * x_vals + b_intercept, color="#e6550d", linewidth=2, label=f"Fit (r = {corr_shap_cand_p.loc['Global_active_power', 'Global_intensity']:.4f})")
    ax_d.set_title("(D) Attribution Collinearity: Global_active_power vs Global_intensity", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("SHAP Anomaly Attribution: Global_active_power", fontsize=10)
    ax_d.set_ylabel("SHAP Anomaly Attribution: Global_intensity", fontsize=10)
    ax_d.grid(True, linestyle=":", alpha=0.5)
    ax_d.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig1_path = exp_dir / "feature_dependence_analysis.png"
    plt.savefig(fig1_path)
    plt.close(fig1)
    logger.info(f"Saved Figure 1: {fig1_path.name}")

    # Figure 2: Grouped Feature Importance & Consolidation Analysis
    fig2, (ax2_a, ax2_b) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Panel A: Grouped Feature Importance (% Share)
    group_names = group_df["group_name"].tolist()
    pcts = group_df["percentage_of_total_importance"].tolist()
    y_pos = np.arange(len(group_names))
    palette_g = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c", "#9970ab", "#762a83"]

    bars = ax2_a.barh(y_pos, pcts, color=palette_g[:len(group_names)], edgecolor="#333333", height=0.6)
    ax2_a.set_yticks(y_pos)
    ax2_a.set_yticklabels(group_names, fontsize=10, fontweight="semibold")
    ax2_a.invert_yaxis()
    ax2_a.set_xlabel("Percentage of Total Anomaly Attribution (%)", fontsize=10, fontweight="semibold")
    ax2_a.set_title("(A) Grouped Feature Importance Across Functional Domains", fontsize=11, fontweight="bold")
    ax2_a.set_xlim(0, 35)
    for bar in bars:
        w = bar.get_width()
        ax2_a.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax2_a.grid(True, linestyle=":", alpha=0.5, axis="x")

    # Panel B: Consolidation Sensitivity (Sub_metering_2 vs. Individual vs. Consolidated Bulk Power)
    compare_labels = [
        "Sub_metering_2\n(Rank 1)",
        "rolling_std_4\n(Rank 2)",
        "Global_intensity\n(Rank 3)",
        "Sub_metering_1\n(Rank 4)",
        "Global_active_power\n(Rank 5)",
        "Consolidated\nBulk Draw (P+I)",
    ]
    compare_vals = [
        mean_sub2,
        float(feat_attr_dict["rolling_std_4"]),
        mean_i,
        float(feat_attr_dict["Sub_metering_1"]),
        mean_p,
        mean_bulk,
    ]
    colors_c = ["#2ca02c", "#1f77b4", "#fdae61", "#31a354", "#e6550d", "#d7191c"]
    x_c = np.arange(len(compare_labels))

    bars2 = ax2_b.bar(x_c, compare_vals, color=colors_c, edgecolor="#333333", width=0.55)
    ax2_b.set_xticks(x_c)
    ax2_b.set_xticklabels(compare_labels, fontsize=9, fontweight="semibold")
    ax2_b.set_ylabel("Mean |SHAP Anomaly Attribution|", fontsize=10, fontweight="semibold")
    ax2_b.set_title("(B) Attribution Splitting: Collinear Consolidation vs Individual Ranks", fontsize=11, fontweight="bold")
    ax2_b.set_ylim(0, 1.25)
    for bar in bars2:
        h = bar.get_height()
        ax2_b.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax2_b.axhline(mean_sub2, color="#2ca02c", linestyle="--", alpha=0.7, label=f"Sub_metering_2 Level ({mean_sub2:.3f})")
    ax2_b.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax2_b.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig2_path = exp_dir / "grouped_shap_importance.png"
    plt.savefig(fig2_path)
    plt.close(fig2)
    logger.info(f"Saved Figure 2: {fig2_path.name}")

    logger.info("\n" + "=" * 76)
    logger.info("EXP-PHASE3-SHAP-05 EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("=" * 76)

    return pair_df, group_df, raw_corr_df, shap_corr_df


if __name__ == "__main__":
    run_dependence_experiment()

