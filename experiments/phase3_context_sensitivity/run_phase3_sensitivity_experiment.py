"""
Phase 3: Experimental Validation of Context-Aware XAI
Experiment 2: Context Matching Sensitivity Analysis
Experiment ID: EXP-PHASE3-CONTEXT-02

Evaluates 6 contextual reference configurations on the identical 333 candidate anomalies:
  Config 1: Hour window +/-0 hr, Season Disabled (Static Hour x Weekend, 48 bins)
  Config 2: Hour window +/-0 hr, Season Enabled  (Diurnal exact hour, Season +/-1 mo)
  Config 3: Hour window +/-1 hr, Season Disabled (Diurnal +/-1 hr, Season Disabled)
  Config 4: Hour window +/-1 hr, Season Enabled  (Baseline Dynamic Twins, +/-1 hr, +/-1 mo)
  Config 5: Hour window +/-2 hr, Season Disabled (Diurnal +/-2 hr, Season Disabled)
  Config 6: Hour window +/-2 hr, Season Enabled  (Diurnal +/-2 hr, Season +/-1 mo)

All configurations strictly reference historical training data (N_train = 109,347)
without lookahead temporal leakage. Fallback threshold: N < 50.
"""

from pathlib import Path
import sys
import os
import logging
import numpy as np
import pandas as pd

# Setup paths
base_dir = Path(__file__).resolve().parent.parent.parent
src_dir = ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir) if ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir)).exists() else base_dir)
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from context_analysis import (
    load_data_splits,
    extract_historical_context_cohort,
)

CONFIG_DEFS = [
    {
        "id": 1,
        "name": "config1_h0_season_off",
        "label": "Hour +/-0h, Season OFF",
        "hour_window": 0,
        "match_season": False,
        "file": "config1_h0_season_off.csv",
    },
    {
        "id": 2,
        "name": "config2_h0_season_on",
        "label": "Hour +/-0h, Season ON",
        "hour_window": 0,
        "match_season": True,
        "file": "config2_h0_season_on.csv",
    },
    {
        "id": 3,
        "name": "config3_h1_season_off",
        "label": "Hour +/-1h, Season OFF",
        "hour_window": 1,
        "match_season": False,
        "file": "config3_h1_season_off.csv",
    },
    {
        "id": 4,
        "name": "config4_h1_season_on",
        "label": "Hour +/-1h, Season ON (Baseline Twins)",
        "hour_window": 1,
        "match_season": True,
        "file": "config4_h1_season_on.csv",
    },
    {
        "id": 5,
        "name": "config5_h2_season_off",
        "label": "Hour +/-2h, Season OFF",
        "hour_window": 2,
        "match_season": False,
        "file": "config5_h2_season_off.csv",
    },
    {
        "id": 6,
        "name": "config6_h2_season_on",
        "label": "Hour +/-2h, Season ON",
        "hour_window": 2,
        "match_season": True,
        "file": "config6_h2_season_on.csv",
    },
]


def run_sensitivity_experiment():
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
    logger = logging.getLogger("EXP-PHASE3-CONTEXT-02")

    logger.info("=" * 76)
    logger.info("MEMBER 4 - PHASE 3: CONTEXT MATCHING SENSITIVITY EXPERIMENT")
    logger.info("Experiment ID: EXP-PHASE3-CONTEXT-02")
    logger.info("=" * 76)

    # 1. Load Data Splits
    features_csv = src_dir / "outputs" / "household_15min_features.csv"
    train_df, test_df = load_data_splits(features_csv, train_fraction=0.8)
    logger.info(f"Loaded train partition: {len(train_df):,} rows | Test partition: {len(test_df):,} rows")

    max_train_ts = train_df["timestamp"].max()
    min_test_ts = test_df["timestamp"].min()
    logger.info(f"Train temporal upper bound: {max_train_ts}")
    logger.info(f"Test temporal lower bound:  {min_test_ts}")
    assert max_train_ts < min_test_ts, "Fatal: Temporal lookahead leakage detected in partition boundaries!"

    # 2. Global training baseline statistics (for extreme fallback)
    global_train_powers = train_df["Global_active_power"].dropna().to_numpy()
    glob_median = float(np.median(global_train_powers))
    glob_mean = float(np.mean(global_train_powers))
    glob_std = float(np.std(global_train_powers))
    logger.info(f"Global training stats: Median={glob_median:.4f} kW, Mean={glob_mean:.4f} kW, Std={glob_std:.4f} kW")

    # 3. Load 333 Candidate Anomaly Timestamps
    cand_csv = src_dir / "outputs" / "contextual_anomalies.csv"
    cand_source_df = pd.read_csv(cand_csv)
    cand_source_df["dt"] = pd.to_datetime(cand_source_df["timestamp"])
    test_df["dt"] = pd.to_datetime(test_df["timestamp"])

    candidate_test_df = test_df[test_df["dt"].isin(cand_source_df["dt"])].copy().reset_index(drop=True)
    assert len(candidate_test_df) == 333, f"Expected 333 candidates, found {len(candidate_test_df)}"
    assert candidate_test_df["dt"].is_unique, "Duplicate candidate timestamps detected!"
    logger.info(f"Aligned {len(candidate_test_df)} candidate anomaly readings.")

    # 4. Evaluate Each Configuration
    config_dfs = {}
    config_stats = {}

    for cfg in CONFIG_DEFS:
        cid = cfg["id"]
        cname = cfg["name"]
        clabel = cfg["label"]
        hw = cfg["hour_window"]
        season = cfg["match_season"]
        fname = cfg["file"]

        logger.info(f"\n--- Running Config {cid}: {clabel} ---")
        records = []
        max_ts_observed = []
        fallback_count = 0
        insufficient_count = 0

        for _, row in candidate_test_df.iterrows():
            ts_str = row["dt"].strftime("%Y-%m-%d %H:%M:%S")
            raw_power = row["Global_active_power"]
            is_missing = bool(pd.isna(raw_power))
            actual_power = np.nan if is_missing else float(raw_power)

            t_hour = int(row["hour"])
            t_weekend = int(row["is_weekend"])
            t_month = int(row["month"])

            # Compute hours
            if hw == 0:
                valid_hours = [t_hour]
            else:
                valid_hours = [(t_hour + d) % 24 for d in range(-hw, hw + 1)]

            hour_mask = train_df["hour"].isin(valid_hours)
            weekend_mask = (train_df["is_weekend"] == t_weekend)

            # Check raw cohort size before seasonal fallback
            fallback_triggered = False
            if season:
                valid_months = [((t_month - 1 + d) % 12) + 1 for d in (-1, 0, 1)]
                season_mask = train_df["month"].isin(valid_months)
                initial_mask = hour_mask & weekend_mask & season_mask
                initial_count = int(np.sum(initial_mask))
                if initial_count < 50:
                    fallback_triggered = True
                    fallback_count += 1
                    effective_mask = hour_mask & weekend_mask
                else:
                    effective_mask = initial_mask
            else:
                effective_mask = hour_mask & weekend_mask

            cohort = train_df[effective_mask]
            cohort_powers = cohort["Global_active_power"].dropna().to_numpy()
            c_size = len(cohort_powers)

            if len(cohort) > 0:
                max_ts_observed.append(cohort["timestamp"].max())

            is_insufficient = c_size < 50
            if is_insufficient:
                insufficient_count += 1

            if c_size == 0:
                c_median = glob_median
                c_mean = glob_mean
                c_std = glob_std
            else:
                c_median = float(np.median(cohort_powers))
                c_mean = float(np.mean(cohort_powers))
                c_std = float(np.std(cohort_powers))

            if is_missing:
                dev = np.nan
                z = np.nan
            else:
                dev = float(abs(actual_power - c_median))
                z = float((actual_power - c_mean) / max(c_std, 1e-6))

            records.append({
                "timestamp": ts_str,
                "actual_power_kw": actual_power,
                "hour_window": hw,
                "match_season": season,
                "cohort_size": c_size,
                "contextual_median_kw": c_median,
                "contextual_mean_kw": c_mean,
                "contextual_std_kw": c_std,
                "contextual_deviation_kw": dev,
                "contextual_z_score": z,
                "is_power_missing": is_missing,
                "is_cohort_insufficient": is_insufficient,
                "fallback_triggered": fallback_triggered,
            })

        # Temporal leakage assertion
        overall_max_ts = max(max_ts_observed)
        assert overall_max_ts <= max_train_ts, f"Leakage: {overall_max_ts} > {max_train_ts}"

        cfg_df = pd.DataFrame(records)
        out_csv = exp_dir / fname
        cfg_df.to_csv(out_csv, index=False)
        config_dfs[cid] = cfg_df

        # Aggregate metrics
        devs = cfg_df["contextual_deviation_kw"].dropna()
        zs = cfg_df["contextual_z_score"].dropna()
        sizes = cfg_df["cohort_size"]
        stds = cfg_df["contextual_std_kw"]

        config_stats[cid] = {
            "name": cname,
            "label": clabel,
            "hour_window": hw,
            "match_season": season,
            "cohort_min": int(sizes.min()),
            "cohort_q25": float(sizes.quantile(0.25)),
            "cohort_median": float(sizes.median()),
            "cohort_mean": float(sizes.mean()),
            "cohort_q75": float(sizes.quantile(0.75)),
            "cohort_max": int(sizes.max()),
            "fallbacks": fallback_count,
            "insufficient": insufficient_count,
            "dev_mean": float(devs.mean()),
            "dev_std": float(devs.std()),
            "dev_min": float(devs.min()),
            "dev_q25": float(devs.quantile(0.25)),
            "dev_median": float(devs.median()),
            "dev_q75": float(devs.quantile(0.75)),
            "dev_max": float(devs.max()),
            "z_mean": float(zs.mean()),
            "z_std": float(zs.std()),
            "z_median": float(zs.median()),
            "z_min": float(zs.min()),
            "z_max": float(zs.max()),
            "std_min": float(stds.min()),
            "std_mean": float(stds.mean()),
            "std_max": float(stds.max()),
            "low_std_count": int(np.sum(stds < 0.20)),
        }

        logger.info(f"  Saved {out_csv.name} (333 records)")
        logger.info(f"  Cohort: Min={sizes.min()}, Median={sizes.median():.0f}, Max={sizes.max()}, Fallbacks={fallback_count}")
        logger.info(f"  Deviation: Mean={devs.mean():.4f} kW, Median={devs.median():.4f} kW, Max={devs.max():.4f} kW")
        logger.info(f"  Z-score: Mean={zs.mean():.4f}, Median={zs.median():.4f}, Max={zs.max():.4f}")

    # 5. Build Combined Sensitivity Comparison CSV
    logger.info("\nBuilding combined_sensitivity_comparison.csv...")
    comb_records = []
    for idx, row in candidate_test_df.iterrows():
        ts_str = row["dt"].strftime("%Y-%m-%d %H:%M:%S")
        actual_power = float(row["Global_active_power"])

        comb_row = {
            "timestamp": ts_str,
            "actual_power_kw": actual_power,
            "hour": int(row["hour"]),
            "day_of_week": int(row.get("day_of_week", row["dt"].weekday())),
            "is_weekend": int(row["is_weekend"]),
            "month": int(row["month"]),
        }

        medians = []
        deviations = []
        stds = []
        z_scores = []

        for cid in range(1, 7):
            c_row = config_dfs[cid].iloc[idx]
            comb_row[f"c{cid}_cohort_size"] = int(c_row["cohort_size"])
            comb_row[f"c{cid}_median_kw"] = float(c_row["contextual_median_kw"])
            comb_row[f"c{cid}_mean_kw"] = float(c_row["contextual_mean_kw"])
            comb_row[f"c{cid}_std_kw"] = float(c_row["contextual_std_kw"])
            comb_row[f"c{cid}_deviation_kw"] = float(c_row["contextual_deviation_kw"])
            comb_row[f"c{cid}_z_score"] = float(c_row["contextual_z_score"])
            comb_row[f"c{cid}_fallback"] = bool(c_row["fallback_triggered"])

            medians.append(c_row["contextual_median_kw"])
            deviations.append(c_row["contextual_deviation_kw"])
            stds.append(c_row["contextual_std_kw"])
            z_scores.append(c_row["contextual_z_score"])

        # Cross-configuration summary metrics
        comb_row["median_min_kw"] = float(np.min(medians))
        comb_row["median_max_kw"] = float(np.max(medians))
        comb_row["median_range_kw"] = float(np.max(medians) - np.min(medians))

        comb_row["deviation_min_kw"] = float(np.min(deviations))
        comb_row["deviation_max_kw"] = float(np.max(deviations))
        comb_row["deviation_range_kw"] = float(np.max(deviations) - np.min(deviations))

        comb_row["std_min_kw"] = float(np.min(stds))
        comb_row["std_max_kw"] = float(np.max(stds))
        comb_row["std_range_kw"] = float(np.max(stds) - np.min(stds))

        comb_row["z_min"] = float(np.min(z_scores))
        comb_row["z_max"] = float(np.max(z_scores))
        comb_row["z_range"] = float(np.max(z_scores) - np.min(z_scores))

        comb_records.append(comb_row)

    comb_df = pd.DataFrame(comb_records)
    comb_csv = exp_dir / "combined_sensitivity_comparison.csv"
    comb_df.to_csv(comb_csv, index=False)
    logger.info(f"Saved combined comparison CSV ({len(comb_df)} rows, {len(comb_df.columns)} cols): {comb_csv.name}")

    logger.info("\n" + "=" * 76)
    logger.info("EXP-PHASE3-CONTEXT-02 SENSITIVITY EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("=" * 76)

    return config_stats, comb_df


if __name__ == "__main__":
    run_sensitivity_experiment()

