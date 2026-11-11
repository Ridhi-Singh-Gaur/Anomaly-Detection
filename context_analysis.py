"""
Context Analysis Module for Household Electricity Anomaly Detection
Part of Member 4: Context-Aware Explainable AI (XAI) Architecture.

This module benchmarks anomalous smart meter observations against historical 
cohorts ('historical twins') extracted strictly from the training period.
It avoids data leakage by ensuring no test or future observations enter the 
context baseline pool.

Targeted Phase 2 Updates:
- Added defensive validation for missing or NaN active power telemetry.
- Explicitly prevented silent NaN propagation in contextual medians, deviations, and z-scores.
- Documented target_day_of_week parameter retention for sensitivity experiments.
- Maintained exact existing twin matching logic without lookahead leakage.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def load_data_splits(
    features_csv_path: Path, 
    train_fraction: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed 15-minute features and perform strict chronological split.
    """
    df = pd.read_csv(features_csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    split_idx = int(len(df) * train_fraction)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    return train_df, test_df


def extract_historical_context_cohort(
    train_df: pd.DataFrame,
    target_hour: int,
    target_day_of_week: int,
    target_is_weekend: int,
    target_month: int,
    hour_window: int = 1,
    match_season: bool = True,
) -> pd.DataFrame:
    """
    Extract a historical context cohort from the training data that matches the
    temporal, diurnal, and seasonal operational context of a target test reading.

    Context matching criteria:
      1. Diurnal: Hours in [target_hour - hour_window, target_hour + hour_window] (modulo 24).
      2. Weekly: Same weekend status (is_weekend == target_is_weekend).
      3. Seasonal: Proximate months in [target_month - 1, target_month + 1] (modulo 12).

    Note on target_day_of_week:
      The target_day_of_week parameter is preserved in the signature for forward API
      compatibility and planned sensitivity experiments (e.g., Sunday-specific vs. weekend
      cohorts), but is intentionally omitted from filtering here to preserve the exact
      verified baseline matching logic and avoid fragmenting historical cohorts.
    """
    # 1. Diurnal hour window
    if hour_window == 0:
        hour_mask = (train_df["hour"] == target_hour)
    else:
        valid_hours = [(target_hour + delta) % 24 for delta in range(-hour_window, hour_window + 1)]
        hour_mask = train_df["hour"].isin(valid_hours)
    
    # 2. Weekly status (weekday vs weekend)
    weekend_mask = (train_df["is_weekend"] == target_is_weekend)
    
    # 3. Seasonal window (adjacent months modulo 12)
    if match_season:
        valid_months = [((target_month - 1 + delta) % 12) + 1 for delta in (-1, 0, 1)]
        season_mask = train_df["month"].isin(valid_months)
        combined_mask = hour_mask & weekend_mask & season_mask
    else:
        combined_mask = hour_mask & weekend_mask
        
    cohort = train_df[combined_mask]
    
    # Fallback to broader cohort if seasonal filtering results in too few samples (< 50)
    if len(cohort) < 50:
        cohort = train_df[hour_mask & weekend_mask]
        
    return cohort


def evaluate_contextual_metrics(
    train_df: pd.DataFrame,
    test_row: pd.Series,
    hour_window: int = 1,
    match_season: bool = True,
) -> Dict[str, Any]:
    """
    Calculate contextual metrics for an individual test observation:
      - Historical expected consumption (mean and median)
      - Absolute deviation from contextual median
      - Contextual percentile rank
      - Contextual z-score
      - Appliance sub-metering contextual profiles

    Safely handles missing or NaN telemetry without crashing or silently corrupting metrics.
    """
    # Defensive handling of active power telemetry
    raw_power = test_row.get("Global_active_power", np.nan)
    if pd.isna(raw_power):
        actual_power = np.nan
        is_power_nan = True
    else:
        actual_power = float(raw_power)
        is_power_nan = False

    t_hour = int(test_row["hour"])
    t_dow = int(test_row.get("day_of_week", 0))
    t_weekend = int(test_row["is_weekend"])
    t_month = int(test_row["month"])
    
    cohort = extract_historical_context_cohort(
        train_df,
        target_hour=t_hour,
        target_day_of_week=t_dow,
        target_is_weekend=t_weekend,
        target_month=t_month,
        hour_window=hour_window,
        match_season=match_season,
    )
    
    cohort_powers = cohort["Global_active_power"].dropna().to_numpy()
    
    if len(cohort_powers) == 0:
        # Extreme fallback to global training median if cohort is entirely empty
        global_train_powers = train_df["Global_active_power"].dropna().to_numpy()
        hist_median = float(np.median(global_train_powers)) if len(global_train_powers) > 0 else 0.0
        hist_mean = float(np.mean(global_train_powers)) if len(global_train_powers) > 0 else 0.0
        hist_std = float(np.std(global_train_powers)) if len(global_train_powers) > 0 else 1.0
        percentile = 50.0 if not is_power_nan else np.nan
    else:
        hist_median = float(np.median(cohort_powers))
        hist_mean = float(np.mean(cohort_powers))
        hist_std = float(np.std(cohort_powers))
        if is_power_nan:
            percentile = np.nan
        else:
            percentile = float(np.mean(cohort_powers <= actual_power) * 100.0)
        
    if is_power_nan:
        abs_deviation = np.nan
        z_score = np.nan
    else:
        abs_deviation = float(abs(actual_power - hist_median))
        z_score = float((actual_power - hist_mean) / max(hist_std, 1e-6))
    
    # Sub-metering contextual metrics (defensively handles missing or NaN columns)
    sub1_raw = test_row.get("Sub_metering_1", 0.0)
    sub2_raw = test_row.get("Sub_metering_2", 0.0)
    sub3_raw = test_row.get("Sub_metering_3", 0.0)

    sub1_actual = 0.0 if pd.isna(sub1_raw) else float(sub1_raw)
    sub2_actual = 0.0 if pd.isna(sub2_raw) else float(sub2_raw)
    sub3_actual = 0.0 if pd.isna(sub3_raw) else float(sub3_raw)
    
    sub1_series = cohort["Sub_metering_1"].dropna() if "Sub_metering_1" in cohort.columns else pd.Series(dtype=float)
    sub2_series = cohort["Sub_metering_2"].dropna() if "Sub_metering_2" in cohort.columns else pd.Series(dtype=float)
    sub3_series = cohort["Sub_metering_3"].dropna() if "Sub_metering_3" in cohort.columns else pd.Series(dtype=float)

    sub1_hist_median = float(sub1_series.median()) if len(sub1_series) > 0 else 0.0
    sub2_hist_median = float(sub2_series.median()) if len(sub2_series) > 0 else 0.0
    sub3_hist_median = float(sub3_series.median()) if len(sub3_series) > 0 else 0.0
    
    return {
        "timestamp": str(test_row["timestamp"]),
        "actual_power_kw": actual_power,
        "context_cohort_size": int(len(cohort)),
        "context_expected_median_kw": hist_median,
        "context_expected_mean_kw": hist_mean,
        "context_std_kw": hist_std,
        "context_abs_deviation_kw": abs_deviation,
        "context_percentile": percentile,
        "context_z_score": z_score,
        "sub1_actual_wh": sub1_actual,
        "sub1_hist_median_wh": sub1_hist_median,
        "sub2_actual_wh": sub2_actual,
        "sub2_hist_median_wh": sub2_hist_median,
        "sub3_actual_wh": sub3_actual,
        "sub3_hist_median_wh": sub3_hist_median,
    }


def batch_context_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_timestamps: list,
) -> pd.DataFrame:
    """
    Run context-aware analysis over a designated list of candidate timestamps.
    Normalizes timestamps via pd.to_datetime to avoid string formatting discrepancies.
    """
    cand_dts = pd.to_datetime(candidate_timestamps)
    test_dts = pd.to_datetime(test_df["timestamp"])
    sub_test = test_df[test_dts.isin(cand_dts)].copy().reset_index(drop=True)
    
    records = []
    for _, row in sub_test.iterrows():
        rec = evaluate_contextual_metrics(train_df, row)
        records.append(rec)
        
    return pd.DataFrame(records)
