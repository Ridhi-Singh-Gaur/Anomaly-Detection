from pathlib import Path
import pandas as pd
import numpy as np

exp_dir = Path("experiments/phase3_context_sensitivity")
comb_df = pd.read_csv(exp_dir / "combined_sensitivity_comparison.csv")

print("=== 1. LARGEST SEASONAL SHIFTS (Season ON vs OFF at h=1: |c4_median - c3_median|) ===")
comb_df["season_shift_h1"] = (comb_df["c4_median_kw"] - comb_df["c3_median_kw"]).abs()
top_season = comb_df.sort_values("season_shift_h1", ascending=False).head(5)
for _, r in top_season.iterrows():
    print(f"{r['timestamp']}: P={r['actual_power_kw']:.3f}kW, H={r['hour']}, Mo={r['month']}, c3_off={r['c3_median_kw']:.3f}kW, c4_on={r['c4_median_kw']:.3f}kW, diff={r['c4_median_kw']-r['c3_median_kw']:+.3f}kW, dev_c3={r['c3_deviation_kw']:.3f}kW, dev_c4={r['c4_deviation_kw']:.3f}kW")

print("\n=== 2. LARGEST WINDOW SHIFTS (h=2 vs h=0 at Season ON: |c6_median - c2_median|) ===")
comb_df["window_shift_season_on"] = (comb_df["c6_median_kw"] - comb_df["c2_median_kw"]).abs()
top_window = comb_df.sort_values("window_shift_season_on", ascending=False).head(5)
for _, r in top_window.iterrows():
    print(f"{r['timestamp']}: P={r['actual_power_kw']:.3f}kW, H={r['hour']}, Mo={r['month']}, c2_h0={r['c2_median_kw']:.3f}kW, c6_h2={r['c6_median_kw']:.3f}kW, diff={r['c6_median_kw']-r['c2_median_kw']:+.3f}kW, dev_c2={r['c2_deviation_kw']:.3f}kW, dev_c6={r['c6_deviation_kw']:.3f}kW")

print("\n=== 3. TOP OVERALL SENSITIVITY CANDIDATES (max median_range_kw) ===")
top_range = comb_df.sort_values("median_range_kw", ascending=False).head(5)
for _, r in top_range.iterrows():
    print(f"{r['timestamp']}: P={r['actual_power_kw']:.3f}kW, H={r['hour']}, Mo={r['month']}, med_range={r['median_range_kw']:.3f}kW (min={r['median_min_kw']:.3f}, max={r['median_max_kw']:.3f}), dev_range={r['deviation_range_kw']:.3f}kW, z_range={r['z_range']:.3f}")

print("\n=== 4. SHOWCASE EVENT (2010-04-11 20:45:00) ACROSS ALL 6 CONFIGS ===")
showcase = comb_df[comb_df["timestamp"] == "2010-04-11 20:45:00"].iloc[0]
for cid in range(1, 7):
    print(f"Config {cid}: Cohort={showcase[f'c{cid}_cohort_size']}, Median={showcase[f'c{cid}_median_kw']:.4f}kW, Mean={showcase[f'c{cid}_mean_kw']:.4f}kW, Std={showcase[f'c{cid}_std_kw']:.4f}kW, Dev={showcase[f'c{cid}_deviation_kw']:.4f}kW, Z={showcase[f'c{cid}_z_score']:.4f}")

