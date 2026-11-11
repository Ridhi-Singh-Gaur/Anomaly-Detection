"""
Generate corrected publication figures for Experiment 5:
EXP-PHASE3-SHAP-05: Feature Dependence and SHAP Attribution Redundancy Audit

Applies scientific unit and terminology corrections:
1. Replaces any implicit or explicit kW references with path-length attribution units.
2. Formally designates bulk draw as Post-Hoc Grouped Attribution phi_bulk = phi_active + phi_intensity.
3. Explicitly annotates the triangle inequality distinction between:
   - Mean absolute grouped attribution: (1/N) sum |phi_P + phi_I| = 1.0540
   - Arithmetic sum of individual means: (1/N) sum |phi_P| + (1/N) sum |phi_I| = 1.0561
4. Designates grouped importance as descriptive model partitioning shares rather than causal physical shares.
5. Preserves original figures while saving *_corrected.png versions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = Path(__file__).resolve().parent.parent.parent
exp_dir = Path(__file__).resolve().parent
shap_file = exp_dir.parent / "phase3_shap_stability" / "shap_values_trees_200.csv"

def generate_corrected_figures():
    # 1. Load data
    raw_corr_df = pd.read_csv(exp_dir / "feature_value_correlations.csv")
    shap_corr_df = pd.read_csv(exp_dir / "shap_attribution_correlations.csv")
    pair_df = pd.read_csv(exp_dir / "feature_correlation_pairs.csv")
    group_df = pd.read_csv(exp_dir / "grouped_shap_importance.csv")
    shap_df = pd.read_csv(shap_file)

    # 333 candidate anomalies
    cand_shap = shap_df.iloc[:333]
    p_phi = cand_shap["shap_Global_active_power"].to_numpy()
    i_phi = cand_shap["shap_Global_intensity"].to_numpy()
    sub2_phi = cand_shap["shap_Sub_metering_2"].to_numpy()
    sub1_phi = cand_shap["shap_Sub_metering_1"].to_numpy()
    std4_phi = cand_shap["shap_rolling_std_4"].to_numpy()

    mean_p = float(np.mean(np.abs(p_phi)))
    mean_i = float(np.mean(np.abs(i_phi)))
    mean_sub2 = float(np.mean(np.abs(sub2_phi)))
    mean_sub1 = float(np.mean(np.abs(sub1_phi)))
    mean_std4 = float(np.mean(np.abs(std4_phi)))

    # Post-hoc grouped bulk attribution
    bulk_phi = p_phi + i_phi
    mean_bulk = float(np.mean(np.abs(bulk_phi)))
    arith_sum_pi = mean_p + mean_i

    # Pivot correlation matrices for candidate anomalies
    cand_raw_pivot = raw_corr_df.pivot(index="feature_1", columns="feature_2", values="candidates_pearson")
    cand_shap_pivot = shap_corr_df.pivot(index="feature_1", columns="feature_2", values="candidates_shap_pearson")

    # -------------------------------------------------------------------------
    # FIGURE 1: Feature Dependence & SHAP Attribution Redundancy (Corrected)
    # -------------------------------------------------------------------------
    fig1, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(15, 13), dpi=300)

    # Panel A: Heatmap of Raw Feature Correlations (Candidates)
    sns.heatmap(cand_raw_pivot, ax=ax_a, cmap="vlag", vmin=-1, vmax=1, center=0,
                annot=False, cbar_kws={"label": "Pearson Correlation (r)"}, square=True)
    ax_a.set_title("(A) Raw Feature Correlations (Candidate Anomalies, N=333)", fontsize=11, fontweight="bold")
    ax_a.tick_params(axis="x", rotation=90, labelsize=8)
    ax_a.tick_params(axis="y", rotation=0, labelsize=8)

    # Panel B: Heatmap of SHAP Attribution Correlations (Candidates)
    sns.heatmap(cand_shap_pivot, ax=ax_b, cmap="vlag", vmin=-1, vmax=1, center=0,
                annot=False, cbar_kws={"label": "SHAP Attribution Correlation (r)"}, square=True)
    ax_b.set_title("(B) TreeSHAP Attribution Correlations (Candidates, N=333)", fontsize=11, fontweight="bold")
    ax_b.tick_params(axis="x", rotation=90, labelsize=8)
    ax_b.tick_params(axis="y", rotation=0, labelsize=8)

    # Panel C: Raw Correlation vs. SHAP Attribution Correlation (Scatter)
    ax_c.scatter(pair_df["cand_raw_pearson"], pair_df["cand_shap_pearson"], color="#1f77b4", alpha=0.7, edgecolors="none", s=40)
    ax_c.plot([-1, 1], [-1, 1], color="#d62728", linestyle="--", linewidth=1.5, label="Identity Line (r_SHAP = r_Raw)")
    
    # Annotate extreme collinear pairs
    p_row = pair_df[((pair_df["feature_1"] == "Global_active_power") & (pair_df["feature_2"] == "Global_intensity"))].iloc[0]
    ax_c.annotate("Global_active_power vs.\nGlobal_intensity (r=0.999, r_SHAP=0.975)",
                  xy=(p_row["cand_raw_pearson"], p_row["cand_shap_pearson"]),
                  xytext=(0.35, 0.82),
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

    # Panel D: Attribution Collinearity: Global_active_power vs Global_intensity (with path-length units)
    ax_d.scatter(p_phi, i_phi, color="#756bb1", alpha=0.7, edgecolors="none", s=35)
    m_slope, b_intercept = np.polyfit(p_phi, i_phi, 1)
    x_vals = np.linspace(min(p_phi), max(p_phi), 100)
    ax_d.plot(x_vals, m_slope * x_vals + b_intercept, color="#e6550d", linewidth=2,
              label=f"Linear Fit (r = {cand_shap_pivot.loc['Global_active_power', 'Global_intensity']:.4f})")
    ax_d.set_title("(D) Attribution Collinearity: Global_active_power vs Global_intensity", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("SHAP Anomaly Attribution: Global_active_power (path-length units)", fontsize=10)
    ax_d.set_ylabel("SHAP Anomaly Attribution: Global_intensity (path-length units)", fontsize=10)
    ax_d.grid(True, linestyle=":", alpha=0.5)
    ax_d.legend(loc="upper left", fontsize=9)
    # Explanatory caption on path length units
    ax_d.text(0.03, 0.04, "Note: Values quantify tree-depth search reduction (path-length units), not kW or A.",
              transform=ax_d.transAxes, fontsize=8, fontstyle="italic", color="#444444",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.8))

    plt.tight_layout()
    fig1_corr_path = exp_dir / "feature_dependence_analysis_corrected.png"
    plt.savefig(fig1_corr_path)
    plt.close(fig1)
    print(f"Saved: {fig1_corr_path.name}")

    # -------------------------------------------------------------------------
    # FIGURE 2: Grouped Importance & Post-Hoc Consolidation (Corrected)
    # -------------------------------------------------------------------------
    fig2, (ax2_a, ax2_b) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Panel A: Descriptive Grouped Feature Importance
    group_names = group_df["group_name"].tolist()
    pcts = group_df["percentage_of_total_importance"].tolist()
    y_pos = np.arange(len(group_names))
    palette_g = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c", "#9970ab", "#762a83"]

    bars = ax2_a.barh(y_pos, pcts, color=palette_g[:len(group_names)], edgecolor="#333333", height=0.6)
    ax2_a.set_yticks(y_pos)
    ax2_a.set_yticklabels(group_names, fontsize=10, fontweight="semibold")
    ax2_a.invert_yaxis()
    ax2_a.set_xlabel("Descriptive Share of Total Anomaly Attribution (%)", fontsize=10, fontweight="semibold")
    ax2_a.set_title("(A) Descriptive Grouped Importance (Ensemble Split Share)", fontsize=11, fontweight="bold")
    ax2_a.set_xlim(0, 35)
    for bar in bars:
        w = bar.get_width()
        ax2_a.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax2_a.grid(True, linestyle=":", alpha=0.5, axis="x")
    ax2_a.text(0.03, 0.04, "Descriptive metric of tree split allocation under collinearity; not physical causal energy shares.",
               transform=ax2_a.transAxes, fontsize=8, fontstyle="italic", color="#444444",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.8))

    # Panel B: Consolidation Analysis with explicit unit and mathematical distinction
    compare_labels = [
        "Sub_metering_2\n(Rank 1)",
        "rolling_std_4\n(Rank 2)",
        "Global_intensity\n(Rank 3)",
        "Sub_metering_1\n(Rank 4)",
        "Global_active_power\n(Rank 5)",
        "Post-Hoc Grouped\nBulk (phi_bulk)",
    ]
    compare_vals = [
        mean_sub2,
        mean_std4,
        mean_i,
        mean_sub1,
        mean_p,
        mean_bulk,
    ]
    colors_c = ["#2ca02c", "#1f77b4", "#fdae61", "#31a354", "#e6550d", "#d7191c"]
    x_c = np.arange(len(compare_labels))

    bars2 = ax2_b.bar(x_c, compare_vals, color=colors_c, edgecolor="#333333", width=0.55)
    ax2_b.set_xticks(x_c)
    ax2_b.set_xticklabels(compare_labels, fontsize=9, fontweight="semibold")
    ax2_b.set_ylabel("Mean |SHAP Attribution| (path-length units)", fontsize=10, fontweight="semibold")
    ax2_b.set_title("(B) Post-Hoc Grouped Attribution vs. Individual Split Attributions", fontsize=11, fontweight="bold")
    ax2_b.set_ylim(0, 1.30)
    for bar in bars2:
        h = bar.get_height()
        ax2_b.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")
    
    # Reference line for Sub_metering_2
    ax2_b.axhline(mean_sub2, color="#2ca02c", linestyle="--", alpha=0.7, label=f"Sub_metering_2 Reference Level ({mean_sub2:.3f})")
    
    # Reference line for arithmetic sum of individual means
    ax2_b.axhline(arith_sum_pi, color="#888888", linestyle=":", linewidth=1.5,
                  label=f"Arithmetic Sum: (1/N)Σ|phi_P| + (1/N)Σ|phi_I| = {arith_sum_pi:.4f}")
    
    # Annotation of triangle inequality divergence
    ax2_b.annotate(f"Triangle inequality discrepancy:\nΔ = 0.0021 (7.5% discordant signs)\nMean |phi_P + phi_I| = {mean_bulk:.4f}",
                   xy=(5, mean_bulk), xytext=(2.6, 1.10),
                   arrowprops=dict(facecolor="#d7191c", shrink=0.08, width=1.2, headwidth=5),
                   fontsize=8, fontweight="medium", color="#990000",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff2f2", edgecolor="#d7191c", alpha=0.9))

    ax2_b.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax2_b.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    fig2_corr_path = exp_dir / "grouped_shap_importance_corrected.png"
    plt.savefig(fig2_corr_path)
    plt.close(fig2)
    print(f"Saved: {fig2_corr_path.name}")

if __name__ == "__main__":
    generate_corrected_figures()
