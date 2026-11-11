"""
Phase 3: Experimental Validation of Context-Aware XAI
Experiment 1: Comparative Evaluation of Three Contextual Reference Strategies

Strategies Evaluated:
  - Strategy A: Global Historical Median (All historical training observations)
  - Strategy B: Hour-Based and Weekend-Based Baseline (diurnal hour x weekend status)
  - Strategy C: Dynamic Historical Twin Matching (Hour +/-1 hr, Weekend status, Month +/-1 mo)

All strategies are evaluated strictly across the identical 333 candidate anomaly timestamps
using training data only (N_train = 109,347) without lookahead leakage.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Path routing
base_dir = Path(__file__).resolve().parent.parent.parent
src_dir = ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir) if ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir)).exists() else base_dir)
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from context_analysis import (
    load_data_splits,
    extract_historical_context_cohort,
)


def run_experiment():
    print("=" * 74)
    print("MEMBER 4 - PHASE 3: CONTEXTUAL REFERENCE STRATEGY EXPERIMENT")
    print("=" * 74)

    exp_dir = Path(__file__).resolve().parent
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data Splits
    features_csv = src_dir / "outputs" / "household_15min_features.csv"
    train_df, test_df = load_data_splits(features_csv, train_fraction=0.8)
    print(f"[Data Split] Train partition: {len(train_df):,} rows | Test partition: {len(test_df):,} rows")
    max_train_ts = train_df["timestamp"].max()
    min_test_ts = test_df["timestamp"].min()
    print(f"  Train max timestamp: {max_train_ts}")
    print(f"  Test min timestamp:  {min_test_ts}")
    assert max_train_ts < min_test_ts, "Fatal: Lookahead temporal leakage in train/test split!"

    # 2. Load 333 Candidate Anomaly Timestamps
    cand_csv = src_dir / "outputs" / "contextual_anomalies.csv"
    cand_source_df = pd.read_csv(cand_csv)
    cand_source_df["dt"] = pd.to_datetime(cand_source_df["timestamp"])
    test_df["dt"] = pd.to_datetime(test_df["timestamp"])

    candidate_test_df = test_df[test_df["dt"].isin(cand_source_df["dt"])].copy().reset_index(drop=True)
    assert len(candidate_test_df) == 333, f"Expected 333 candidates, found {len(candidate_test_df)}"
    assert candidate_test_df["dt"].is_unique, "Fatal: Duplicate candidate timestamps detected!"
    print(f"[Candidates] Successfully aligned all {len(candidate_test_df)} candidate anomaly readings.")

    # --------------------------------------------------------------------------
    # STRATEGY A: Global Historical Median
    # --------------------------------------------------------------------------
    print("\n[Strategy A] Evaluating Global Historical Baseline...")
    global_powers = train_df["Global_active_power"].dropna().to_numpy()
    glob_cohort_size = len(global_powers)
    glob_median = float(np.median(global_powers))
    glob_mean = float(np.mean(global_powers))
    glob_std = float(np.std(global_powers))

    records_a = []
    for _, row in candidate_test_df.iterrows():
        ts_str = row["dt"].strftime("%Y-%m-%d %H:%M:%S")
        raw_power = row["Global_active_power"]
        is_missing = bool(pd.isna(raw_power))
        actual_power = np.nan if is_missing else float(raw_power)

        if is_missing:
            dev = np.nan
            z = np.nan
        else:
            dev = float(abs(actual_power - glob_median))
            z = float((actual_power - glob_mean) / max(glob_std, 1e-6))

        records_a.append({
            "timestamp": ts_str,
            "actual_power_kw": actual_power,
            "cohort_size": glob_cohort_size,
            "contextual_median_kw": glob_median,
            "contextual_mean_kw": glob_mean,
            "contextual_std_kw": glob_std,
            "contextual_deviation_kw": dev,
            "contextual_z_score": z,
            "is_power_missing": is_missing,
            "is_cohort_insufficient": False,
        })

    df_a = pd.DataFrame(records_a)
    out_a = exp_dir / "strategy_a_global.csv"
    df_a.to_csv(out_a, index=False)
    print(f"  Saved Strategy A CSV ({len(df_a)} rows): {out_a.name}")

    # --------------------------------------------------------------------------
    # STRATEGY B: Hour-Based and Weekend-Based Baseline
    # --------------------------------------------------------------------------
    print("\n[Strategy B] Evaluating Hour-and-Weekend Baseline (48 static bins)...")
    records_b = []
    # Pre-index training data by hour and is_weekend for fast, exact cohort retrieval
    hw_groups = train_df.groupby(["hour", "is_weekend"])

    for _, row in candidate_test_df.iterrows():
        ts_str = row["dt"].strftime("%Y-%m-%d %H:%M:%S")
        raw_power = row["Global_active_power"]
        is_missing = bool(pd.isna(raw_power))
        actual_power = np.nan if is_missing else float(raw_power)

        h_val = int(row["hour"])
        w_val = int(row["is_weekend"])

        if (h_val, w_val) in hw_groups.groups:
            cohort_indices = hw_groups.groups[(h_val, w_val)]
            b_powers = train_df.loc[cohort_indices, "Global_active_power"].dropna().to_numpy()
        else:
            b_powers = np.array([])

        b_cohort_size = len(b_powers)
        is_insufficient = b_cohort_size < 50

        if b_cohort_size == 0:
            b_median = glob_median
            b_mean = glob_mean
            b_std = glob_std
        else:
            b_median = float(np.median(b_powers))
            b_mean = float(np.mean(b_powers))
            b_std = float(np.std(b_powers))

        if is_missing:
            dev = np.nan
            z = np.nan
        else:
            dev = float(abs(actual_power - b_median))
            z = float((actual_power - b_mean) / max(b_std, 1e-6))

        records_b.append({
            "timestamp": ts_str,
            "actual_power_kw": actual_power,
            "cohort_size": b_cohort_size,
            "contextual_median_kw": b_median,
            "contextual_mean_kw": b_mean,
            "contextual_std_kw": b_std,
            "contextual_deviation_kw": dev,
            "contextual_z_score": z,
            "is_power_missing": is_missing,
            "is_cohort_insufficient": is_insufficient,
        })

    df_b = pd.DataFrame(records_b)
    out_b = exp_dir / "strategy_b_hour_weekend.csv"
    df_b.to_csv(out_b, index=False)
    print(f"  Saved Strategy B CSV ({len(df_b)} rows): {out_b.name}")

    # --------------------------------------------------------------------------
    # STRATEGY C: Dynamic Historical Twin Matching
    # --------------------------------------------------------------------------
    print("\n[Strategy C] Evaluating Dynamic Historical Twin Matching (Diurnal +/-1hr, Weekend, Season +/-1mo)...")
    records_c = []
    max_twin_timestamps = []

    for _, row in candidate_test_df.iterrows():
        ts_str = row["dt"].strftime("%Y-%m-%d %H:%M:%S")
        raw_power = row["Global_active_power"]
        is_missing = bool(pd.isna(raw_power))
        actual_power = np.nan if is_missing else float(raw_power)

        c_cohort = extract_historical_context_cohort(
            train_df,
            target_hour=int(row["hour"]),
            target_day_of_week=int(row.get("day_of_week", 0)),
            target_is_weekend=int(row["is_weekend"]),
            target_month=int(row["month"]),
            hour_window=1,
            match_season=True,
        )

        c_powers = c_cohort["Global_active_power"].dropna().to_numpy()
        c_cohort_size = len(c_powers)
        is_insufficient = c_cohort_size < 50

        if len(c_cohort) > 0:
            max_twin_timestamps.append(c_cohort["timestamp"].max())

        if c_cohort_size == 0:
            c_median = glob_median
            c_mean = glob_mean
            c_std = glob_std
        else:
            c_median = float(np.median(c_powers))
            c_mean = float(np.mean(c_powers))
            c_std = float(np.std(c_powers))

        if is_missing:
            dev = np.nan
            z = np.nan
        else:
            dev = float(abs(actual_power - c_median))
            z = float((actual_power - c_mean) / max(c_std, 1e-6))

        records_c.append({
            "timestamp": ts_str,
            "actual_power_kw": actual_power,
            "cohort_size": c_cohort_size,
            "contextual_median_kw": c_median,
            "contextual_mean_kw": c_mean,
            "contextual_std_kw": c_std,
            "contextual_deviation_kw": dev,
            "contextual_z_score": z,
            "is_power_missing": is_missing,
            "is_cohort_insufficient": is_insufficient,
        })

    # Assert zero lookahead leakage
    assert max(max_twin_timestamps) <= max_train_ts, "Fatal: Lookahead leakage detected in Strategy C!"

    df_c = pd.DataFrame(records_c)
    out_c = exp_dir / "strategy_c_dynamic_twins.csv"
    df_c.to_csv(out_c, index=False)
    print(f"  Saved Strategy C CSV ({len(df_c)} rows): {out_c.name}")

    # --------------------------------------------------------------------------
    # COMBINED COMPARISON TABLE
    # --------------------------------------------------------------------------
    print("\n[Comparison] Building Combined Strategy Comparison Dataset...")
    combined_rows = []
    night_hours = {22, 23, 0, 1, 2, 3, 4, 5, 6}

    for i, row in candidate_test_df.iterrows():
        ts_str = row["dt"].strftime("%Y-%m-%d %H:%M:%S")
        hour_val = int(row["hour"])
        is_weekend_val = int(row["is_weekend"])
        month_val = int(row["month"])
        day_of_week_val = int(row.get("day_of_week", 0))
        is_night = hour_val in night_hours

        rec_a = records_a[i]
        rec_b = records_b[i]
        rec_c = records_c[i]

        dev_a = rec_a["contextual_deviation_kw"]
        dev_b = rec_b["contextual_deviation_kw"]
        dev_c = rec_c["contextual_deviation_kw"]

        med_a = rec_a["contextual_median_kw"]
        med_b = rec_b["contextual_median_kw"]
        med_c = rec_c["contextual_median_kw"]

        diff_dev_c_vs_a = dev_c - dev_a
        diff_dev_c_vs_b = dev_c - dev_b
        diff_med_c_vs_b = med_c - med_b

        combined_rows.append({
            "timestamp": ts_str,
            "actual_power_kw": rec_c["actual_power_kw"],
            "hour": hour_val,
            "day_of_week": day_of_week_val,
            "is_weekend": is_weekend_val,
            "month": month_val,
            "is_nighttime": is_night,
            # Strategy A
            "global_cohort_size": rec_a["cohort_size"],
            "global_median_kw": med_a,
            "global_mean_kw": rec_a["contextual_mean_kw"],
            "global_std_kw": rec_a["contextual_std_kw"],
            "global_deviation_kw": dev_a,
            "global_z_score": rec_a["contextual_z_score"],
            # Strategy B
            "hw_cohort_size": rec_b["cohort_size"],
            "hw_median_kw": med_b,
            "hw_mean_kw": rec_b["contextual_mean_kw"],
            "hw_std_kw": rec_b["contextual_std_kw"],
            "hw_deviation_kw": dev_b,
            "hw_z_score": rec_b["contextual_z_score"],
            # Strategy C
            "dynamic_cohort_size": rec_c["cohort_size"],
            "dynamic_median_kw": med_c,
            "dynamic_mean_kw": rec_c["contextual_mean_kw"],
            "dynamic_std_kw": rec_c["contextual_std_kw"],
            "dynamic_deviation_kw": dev_c,
            "dynamic_z_score": rec_c["contextual_z_score"],
            # Comparative Differentials
            "diff_deviation_dynamic_vs_global": diff_dev_c_vs_a,
            "diff_deviation_dynamic_vs_hw": diff_dev_c_vs_b,
            "diff_median_dynamic_vs_hw": diff_med_c_vs_b,
        })

    df_comb = pd.DataFrame(combined_rows)
    out_comb = exp_dir / "combined_strategy_comparison.csv"
    df_comb.to_csv(out_comb, index=False)
    print(f"  Saved Combined Comparison CSV ({len(df_comb)} rows, {len(df_comb.columns)} cols): {out_comb.name}")

    # --------------------------------------------------------------------------
    # SUMMARY STATISTICS COMPUTATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("EXPERIMENTAL SUMMARY RESULTS")
    print("=" * 74)

    def stats_summary(s: pd.Series):
        return {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "min": float(s.min()),
            "25%": float(s.quantile(0.25)),
            "75%": float(s.quantile(0.75)),
            "max": float(s.max()),
        }

    print("\n1. CONTEXTUAL DEVIATIONS (|P_t - median| in kW):")
    dev_summary = pd.DataFrame({
        "Strategy A (Global)": stats_summary(df_comb["global_deviation_kw"]),
        "Strategy B (Hour-Weekend)": stats_summary(df_comb["hw_deviation_kw"]),
        "Strategy C (Dynamic Twins)": stats_summary(df_comb["dynamic_deviation_kw"]),
    })
    print(dev_summary.round(4))

    print("\n2. COHORT SIZES (Observations N):")
    cohort_summary = pd.DataFrame({
        "Strategy A (Global)": stats_summary(df_comb["global_cohort_size"]),
        "Strategy B (Hour-Weekend)": stats_summary(df_comb["hw_cohort_size"]),
        "Strategy C (Dynamic Twins)": stats_summary(df_comb["dynamic_cohort_size"]),
    })
    print(cohort_summary.round(1))

    print("\n3. DIURNAL COMPARISON: NIGHTTIME (N = {}) vs DAYTIME (N = {}):".format(
        df_comb["is_nighttime"].sum(), (~df_comb["is_nighttime"]).sum()
    ))
    night_df = df_comb[df_comb["is_nighttime"]]
    day_df = df_comb[~df_comb["is_nighttime"]]

    diurnal_comp = pd.DataFrame({
        "Night_Dev_Global": stats_summary(night_df["global_deviation_kw"]),
        "Night_Dev_HW": stats_summary(night_df["hw_deviation_kw"]),
        "Night_Dev_Dynamic": stats_summary(night_df["dynamic_deviation_kw"]),
        "Day_Dev_Global": stats_summary(day_df["global_deviation_kw"]),
        "Day_Dev_HW": stats_summary(day_df["hw_deviation_kw"]),
        "Day_Dev_Dynamic": stats_summary(day_df["dynamic_deviation_kw"]),
    })
    print(diurnal_comp[["Night_Dev_Global", "Night_Dev_HW", "Night_Dev_Dynamic"]].round(4))
    print(diurnal_comp[["Day_Dev_Global", "Day_Dev_HW", "Day_Dev_Dynamic"]].round(4))

    print("\n4. WEEKLY COMPARISON: WEEKDAY (N = {}) vs WEEKEND (N = {}):".format(
        (df_comb["is_weekend"] == 0).sum(), (df_comb["is_weekend"] == 1).sum()
    ))
    wday_df = df_comb[df_comb["is_weekend"] == 0]
    wend_df = df_comb[df_comb["is_weekend"] == 1]

    weekly_comp = pd.DataFrame({
        "Weekday_Dev_Global": stats_summary(wday_df["global_deviation_kw"]),
        "Weekday_Dev_HW": stats_summary(wday_df["hw_deviation_kw"]),
        "Weekday_Dev_Dynamic": stats_summary(wday_df["dynamic_deviation_kw"]),
        "Weekend_Dev_Global": stats_summary(wend_df["global_deviation_kw"]),
        "Weekend_Dev_HW": stats_summary(wend_df["hw_deviation_kw"]),
        "Weekend_Dev_Dynamic": stats_summary(wend_df["dynamic_deviation_kw"]),
    })
    print(weekly_comp[["Weekday_Dev_Global", "Weekday_Dev_HW", "Weekday_Dev_Dynamic"]].round(4))
    print(weekly_comp[["Weekend_Dev_Global", "Weekend_Dev_HW", "Weekend_Dev_Dynamic"]].round(4))

    print("\n5. TOP 5 DIVERGENT CANDIDATES (Dynamic vs Static HW):")
    df_comb["abs_diff_dev_c_vs_b"] = df_comb["diff_deviation_dynamic_vs_hw"].abs()
    top_div = df_comb.sort_values("abs_diff_dev_c_vs_b", ascending=False).head(5)
    print(top_div[[
        "timestamp", "actual_power_kw", "hw_median_kw", "dynamic_median_kw",
        "hw_deviation_kw", "dynamic_deviation_kw", "diff_deviation_dynamic_vs_hw"
    ]].to_string(index=False))

    print("\n" + "=" * 74)
    print("PHASE 3 EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 74)


if __name__ == "__main__":
    run_experiment()

