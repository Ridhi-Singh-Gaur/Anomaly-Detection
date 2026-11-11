"""
Regression Test Suite for Member 4 XAI Pipeline
Validates Phase 2 code quality, reliability corrections, and scientific guardrails.

Tests included:
1. Missing context dictionary keys.
2. NaN active power handling.
3. Empty SHAP feature lists.
4. Four-model consensus wording ('Four-Model Consensus Candidate Anomaly').
5. Unsupported appliance names not appearing in explanations.
6. Historical cohort extraction using training data only (zero test data leakage).
7. CSV export containing context_cohort_size column.
8. Exact preservation of all 333 candidate anomaly records.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add source directory to path
base_dir = Path(__file__).resolve().parent.parent
src_dir = ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir) if ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir)).exists() else base_dir)
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from explanation_generator import generate_single_explanation
from context_analysis import (
    load_data_splits,
    extract_historical_context_cohort,
    evaluate_contextual_metrics,
)


# ==============================================================================
# TEST 1: Missing context dictionary keys
# ==============================================================================
def test_missing_context_dictionary_keys():
    """Verify generate_single_explanation does not raise KeyError when context keys are missing."""
    sparse_context = {
        "context_expected_median_kw": 1.5,
        # Omit: context_expected_mean_kw, context_abs_deviation_kw, percentile, z_score, sub-metering
    }
    flags = {"m1": True, "m2": True}
    scores = {"m1": 0.5, "m2": 0.6}
    shap_feats = [{"feature": "GAP", "feature_value": 3.0, "attribution": 0.8}]

    expl = generate_single_explanation(
        timestamp="2010-04-11 20:45:00",
        actual_power=3.5,
        context_metrics=sparse_context,
        model_flags=flags,
        model_scores=scores,
        top_shap_features=shap_feats,
    )
    assert expl is not None
    assert "human_readable_diagnostic" in expl
    assert expl["consensus_level"] == "Two-Model Agreement"
    assert "Active power of 3.50 kW" in expl["human_readable_diagnostic"]


# ==============================================================================
# TEST 2: NaN active power handling
# ==============================================================================
def test_nan_active_power():
    """Verify that NaN active power is handled safely without crashing or corrupting text."""
    valid_context = {
        "context_expected_median_kw": 1.5,
        "context_expected_mean_kw": 1.6,
        "context_abs_deviation_kw": 0.0,
        "context_percentile": 50.0,
        "context_z_score": 0.0,
    }
    flags = {"m1": True}
    scores = {"m1": 0.5}

    # Case A: float('nan')
    expl_nan = generate_single_explanation(
        timestamp="2010-04-11 20:45:00",
        actual_power=float("nan"),
        context_metrics=valid_context,
        model_flags=flags,
        model_scores=scores,
        top_shap_features=[],
    )
    assert math.isnan(expl_nan["actual_power_kw"])
    assert "Active power reading is missing or NaN" in expl_nan["human_readable_diagnostic"]

    # Case B: Context metric evaluation with NaN power
    features_csv = src_dir / "outputs" / "household_15min_features.csv"
    train_df, test_df = load_data_splits(features_csv)
    test_row = test_df.iloc[0].copy()
    test_row["Global_active_power"] = np.nan

    c_eval = evaluate_contextual_metrics(train_df, test_row)
    assert math.isnan(c_eval["actual_power_kw"])
    assert math.isnan(c_eval["context_abs_deviation_kw"])
    assert math.isnan(c_eval["context_percentile"])
    assert math.isnan(c_eval["context_z_score"])
    assert not math.isnan(c_eval["context_expected_median_kw"])  # Baseline median remains valid


# ==============================================================================
# TEST 3: Empty SHAP feature lists
# ==============================================================================
def test_empty_shap_feature_lists():
    """Verify that an empty top_shap_features list produces clean text without dangling punctuation."""
    context = {
        "context_expected_median_kw": 1.5,
        "context_expected_mean_kw": 1.6,
        "context_abs_deviation_kw": 1.5,
        "context_percentile": 99.0,
        "context_z_score": 3.0,
    }
    expl = generate_single_explanation(
        timestamp="2010-04-11 20:45:00",
        actual_power=3.0,
        context_metrics=context,
        model_flags={"m1": True},
        model_scores={"m1": 0.5},
        top_shap_features=[],
    )
    diag = expl["human_readable_diagnostic"]
    assert "No dominant feature attributions identified." in diag
    assert "Key driving features: ." not in diag
    assert not diag.endswith(" : .")


# ==============================================================================
# TEST 4: Four-model consensus wording
# ==============================================================================
def test_four_model_consensus_wording():
    """Verify consensus level is exactly 'Four-Model Consensus Candidate Anomaly'."""
    four_flags = {
        "Statistical_Baseline": True,
        "Isolation_Forest": True,
        "LSTM_Prediction": True,
        "LSTM_Autoencoder": True,
    }
    context = {
        "context_expected_median_kw": 1.99,
        "context_abs_deviation_kw": 5.10,
        "context_percentile": 99.8,
    }
    expl = generate_single_explanation(
        timestamp="2010-04-11 20:45:00",
        actual_power=7.09,
        context_metrics=context,
        model_flags=four_flags,
        model_scores={},
        top_shap_features=[],
    )
    assert expl["consensus_level"] == "Four-Model Consensus Candidate Anomaly"
    assert "High-Confidence" not in expl["consensus_level"]
    assert "Four-Model Consensus Candidate Anomaly" in expl["human_readable_diagnostic"]


# ==============================================================================
# TEST 5: Unsupported appliance names not appearing in explanations
# ==============================================================================
def test_unsupported_appliance_names_absent():
    """Verify no appliance hallucination appears in explanations even under high sub-metering surges."""
    context = {
        "context_expected_median_kw": 1.0,
        "context_abs_deviation_kw": 4.0,
        "context_percentile": 99.9,
        "context_z_score": 4.0,
        "sub1_actual_wh": 100.0,
        "sub1_hist_median_wh": 0.0,
        "sub2_actual_wh": 100.0,
        "sub2_hist_median_wh": 0.0,
        "sub3_actual_wh": 100.0,
        "sub3_hist_median_wh": 0.0,
    }
    expl = generate_single_explanation(
        timestamp="2010-04-11 20:45:00",
        actual_power=5.0,
        context_metrics=context,
        model_flags={"m1": True},
        model_scores={},
        top_shap_features=[],
    )
    diag = expl["human_readable_diagnostic"].lower()

    # Prohibited ungrounded terms
    prohibited_terms = [
        "microwave",
        "dishwasher",
        "oven",
        "washing machine",
        "dryer",
        "refrigerator",
        "compressor",
    ]
    for term in prohibited_terms:
        assert term not in diag, f"Forbidden appliance term '{term}' detected in diagnostic text!"

    # Allowed circuit-level terms
    assert "kitchen sub-circuit" in diag
    assert "laundry-room sub-circuit" in diag
    assert "water heater / air-conditioning sub-circuit" in diag


# ==============================================================================
# TEST 6: Historical cohort extraction using training data only
# ==============================================================================
def test_historical_cohort_training_data_only():
    """Verify historical cohort extraction does not leak any test or future data."""
    features_csv = src_dir / "outputs" / "household_15min_features.csv"
    train_df, test_df = load_data_splits(features_csv)

    max_train_ts = train_df["timestamp"].max()
    min_test_ts = test_df["timestamp"].min()
    assert max_train_ts < min_test_ts, "Train and test partitions overlap in time!"

    # Extract cohort for a test reading
    test_row = test_df.iloc[100]
    cohort = extract_historical_context_cohort(
        train_df,
        target_hour=int(test_row["hour"]),
        target_day_of_week=int(test_row["day_of_week"]),
        target_is_weekend=int(test_row["is_weekend"]),
        target_month=int(test_row["month"]),
        hour_window=1,
        match_season=True,
    )

    # Assert cohort timestamps are all strictly within train_df
    assert len(cohort) > 0
    assert cohort["timestamp"].max() <= max_train_ts
    assert (cohort["timestamp"] >= min_test_ts).sum() == 0, "Test data leakage detected in historical cohort!"


# ==============================================================================
# TEST 7: CSV export containing context_cohort_size column
# ==============================================================================
def test_csv_export_contains_context_cohort_size():
    """Verify that contextual_anomalies.csv includes context_cohort_size with valid positive integers."""
    csv_path = src_dir / "outputs" / "contextual_anomalies.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    df = pd.read_csv(csv_path)

    assert "context_cohort_size" in df.columns, "context_cohort_size column missing from CSV export!"
    assert (df["context_cohort_size"] > 0).all(), "Found non-positive cohort sizes in CSV export!"
    assert df["context_cohort_size"].dtype in [np.int64, int], "context_cohort_size is not integer typed!"


# ==============================================================================
# TEST 8: Exact preservation of all 333 candidate anomaly records
# ==============================================================================
def test_preservation_of_all_333_candidate_records():
    """Verify that exactly 333 candidate records exist without duplicates or missing values."""
    csv_path = src_dir / "outputs" / "contextual_anomalies.csv"
    df = pd.read_csv(csv_path)

    assert len(df) == 333, f"Expected 333 candidate records, got {len(df)}"
    assert df["timestamp"].is_unique, "Found duplicate timestamps in contextual_anomalies.csv!"
    assert df.isnull().sum().sum() == 0, "Found unexpected null values in contextual_anomalies.csv!"

    # Check model agreement breakdown
    count_4 = (df["models_flagged_count"] == 4).sum()
    count_3 = (df["models_flagged_count"] == 3).sum()
    count_2 = (df["models_flagged_count"] == 2).sum()
    count_1 = (df["models_flagged_count"] == 1).sum()

    assert count_4 == 4, f"Expected 4 four-model consensus events, got {count_4}"
    assert count_3 == 37, f"Expected 37 three-model consensus events, got {count_3}"
    assert count_2 == 50, f"Expected 50 two-model agreement events, got {count_2}"
    assert count_1 == 242, f"Expected 242 single-model alert events, got {count_1}"
    assert count_4 + count_3 + count_2 + count_1 == 333


if __name__ == "__main__":
    pytest.main(["-v", __file__])

