"""
Explanation Generator Module for Household Electricity Anomaly Detection
Part of Member 4: Context-Aware Explainable AI (XAI) Architecture.

This module converts quantitative anomaly scores, contextual deviations,
sub-metering profiles, and game-theoretic SHAP attributions into deterministic,
human-understandable diagnostic summaries.

Targeted Phase 2 Updates:
- Aligned 4-model consensus label to 'Four-Model Consensus Candidate Anomaly'.
- Removed ungrounded appliance-specific claims (microwave, dishwasher, dryer, compressor).
- Restricted attribution language strictly to circuit-level telemetry (Sub_metering_1, 2, 3).
- Added robust handling for missing keys, empty SHAP lists, and NaN values.
"""

import math
from typing import Dict, Any, List
import json
import pandas as pd


def generate_single_explanation(
    timestamp: str,
    actual_power: float,
    context_metrics: Dict[str, Any],
    model_flags: Dict[str, bool],
    model_scores: Dict[str, float],
    top_shap_features: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a structured, human-understandable explanation for an anomalous timestamp.
    Safely handles missing keys, empty attribution lists, and NaN telemetry values.
    """
    # 1. Defensive retrieval of contextual metrics with safe defaults
    hist_median = context_metrics.get("context_expected_median_kw", 0.0)
    hist_mean = context_metrics.get("context_expected_mean_kw", 0.0)
    abs_dev = context_metrics.get("context_abs_deviation_kw", 0.0)
    percentile = context_metrics.get("context_percentile", 50.0)
    z_score = context_metrics.get("context_z_score", 0.0)

    # Cast and check NaNs in context metrics
    hist_median = float(hist_median) if hist_median is not None and not pd.isna(hist_median) else 0.0
    hist_mean = float(hist_mean) if hist_mean is not None and not pd.isna(hist_mean) else 0.0
    abs_dev = float(abs_dev) if abs_dev is not None and not pd.isna(abs_dev) else 0.0
    percentile = float(percentile) if percentile is not None and not pd.isna(percentile) else 50.0
    z_score = float(z_score) if z_score is not None and not pd.isna(z_score) else 0.0

    # 2. Defensive handling of actual power reading
    is_power_nan = actual_power is None or pd.isna(actual_power)
    if is_power_nan:
        actual_power_val = float("nan")
        actual_power_str = "N/A"
    else:
        actual_power_val = float(actual_power)
        actual_power_str = f"{actual_power_val:.2f} kW"

    # 3. Sub-metering indicators (telemetry values and medians)
    sub1 = float(context_metrics.get("sub1_actual_wh", 0.0) or 0.0)
    sub2 = float(context_metrics.get("sub2_actual_wh", 0.0) or 0.0)
    sub3 = float(context_metrics.get("sub3_actual_wh", 0.0) or 0.0)
    if math.isnan(sub1): sub1 = 0.0
    if math.isnan(sub2): sub2 = 0.0
    if math.isnan(sub3): sub3 = 0.0

    sub1_med = float(context_metrics.get("sub1_hist_median_wh", 0.0) or 0.0)
    sub2_med = float(context_metrics.get("sub2_hist_median_wh", 0.0) or 0.0)
    sub3_med = float(context_metrics.get("sub3_hist_median_wh", 0.0) or 0.0)
    if math.isnan(sub1_med): sub1_med = 0.0
    if math.isnan(sub2_med): sub2_med = 0.0
    if math.isnan(sub3_med): sub3_med = 0.0

    # 4. Multi-Model Consensus level classification (aligned to conference paper)
    num_models_flagged = sum(1 for v in model_flags.values() if v)
    if num_models_flagged == 4:
        consensus_level = "Four-Model Consensus Candidate Anomaly"
    elif num_models_flagged == 3:
        consensus_level = "Three-Way Multi-Model Consensus"
    elif num_models_flagged == 2:
        consensus_level = "Two-Model Agreement"
    elif num_models_flagged == 1:
        flagged_model_names = [k for k, v in model_flags.items() if v]
        consensus_level = f"Isolated Flag ({', '.join(flagged_model_names)})"
    else:
        consensus_level = "Zero-Model Alert (Non-Candidate)"

    # 5. Contextual verdict text
    if is_power_nan:
        context_summary = "Active power reading is missing or NaN; contextual baseline comparison cannot be evaluated."
    elif actual_power_val > hist_median:
        direction = "above"
        ratio = (actual_power_val / max(hist_median, 0.01))
        context_summary = (
            f"Active power of {actual_power_val:.2f} kW was {abs_dev:.2f} kW {direction} "
            f"the historical contextual median ({hist_median:.2f} kW), representing "
            f"{ratio:.2f}x typical consumption for this temporal regime (Percentile: {percentile:.1f}%)."
        )
    else:
        direction = "below"
        context_summary = (
            f"Active power of {actual_power_val:.2f} kW was {abs_dev:.2f} kW {direction} "
            f"the historical contextual median ({hist_median:.2f} kW, Percentile: {percentile:.1f}%)."
        )

    # 6. Sub-metering supporting clues (Strict adherence: circuit telemetry, not specific unmeasured appliances)
    submeter_clues = []
    if sub1 > max(10.0, sub1_med * 3.0):
        submeter_clues.append(
            f"Elevated Sub-metering 1 reading ({sub1:.1f} Wh vs. contextual median {sub1_med:.1f} Wh) "
            f"indicates localized load on the kitchen sub-circuit."
        )
    if sub2 > max(10.0, sub2_med * 3.0):
        submeter_clues.append(
            f"Elevated Sub-metering 2 reading ({sub2:.1f} Wh vs. contextual median {sub2_med:.1f} Wh) "
            f"indicates localized load on the laundry-room sub-circuit."
        )
    if sub3 > max(10.0, sub3_med * 2.0):
        submeter_clues.append(
            f"Elevated Sub-metering 3 reading ({sub3:.1f} Wh vs. contextual median {sub3_med:.1f} Wh) "
            f"indicates localized load on the water heater / air-conditioning sub-circuit."
        )

    if not submeter_clues:
        submeter_clues.append(
            f"Sub-metering readings (Sub1: {sub1:.1f} Wh, Sub2: {sub2:.1f} Wh, Sub3: {sub3:.1f} Wh) "
            f"were relatively low compared with the observed household load. "
            f"The available measurements do not identify the source of the remaining consumption."
        )

    # 7. SHAP feature attribution summary (distinguishes severity, normalcy, and negligible impact)
    pos_clues = []
    neg_clues = []
    negligible_clues = []

    if top_shap_features:
        for f in top_shap_features[:3]:
            feat_name = f.get("feature", "unknown")
            raw_val = f.get("feature_value", 0.0)
            attribution = f.get("attribution", 0.0)

            val_str = "N/A" if raw_val is None or pd.isna(raw_val) else f"{float(raw_val):.2f}"
            attr_val = 0.0 if attribution is None or pd.isna(attribution) else float(attribution)
            attr_str = f"{attr_val:+.2f}"

            if abs(attr_val) < 0.05:
                negligible_clues.append(f"{feat_name} = {val_str} (SHAP: {attr_str})")
            elif attr_val >= 0.05:
                pos_clues.append(f"{feat_name} = {val_str} pushed anomaly score higher (SHAP: {attr_str})")
            else:
                neg_clues.append(f"{feat_name} = {val_str} acted toward normalcy (SHAP: {attr_str})")

    shap_statements = []
    if pos_clues:
        shap_statements.append(f"Features contributing towards anomaly severity: {'; '.join(pos_clues)}.")
    if neg_clues:
        shap_statements.append(f"Features contributing towards normality: {'; '.join(neg_clues)}.")
    if negligible_clues:
        shap_statements.append(f"Features with negligible attribution: {'; '.join(negligible_clues)}.")

    if shap_statements:
        shap_summary_text = " ".join(shap_statements)
    else:
        shap_summary_text = "No dominant feature attributions identified."

    # 8. Formulate synthesized human-readable diagnostic text
    synthesized_text = (
        f"Event at {timestamp}: {consensus_level}. "
        f"{context_summary} "
        f"Sub-circuit telemetry: {' '.join(submeter_clues)} "
        f"{shap_summary_text}"
    )

    return {
        "timestamp": timestamp,
        "actual_power_kw": actual_power_val,
        "consensus_level": consensus_level,
        "num_models_flagged": num_models_flagged,
        "model_flags": model_flags,
        "model_scores": model_scores,
        "context_evaluation": {
            "historical_expected_median_kw": hist_median,
            "historical_expected_mean_kw": hist_mean,
            "absolute_deviation_kw": abs_dev,
            "contextual_percentile": percentile,
            "contextual_z_score": z_score,
        },
        "submetering_supporting_clues": submeter_clues,
        "top_shap_features": top_shap_features,
        "human_readable_diagnostic": synthesized_text,
    }


def save_explanations_to_json(explanations: List[Dict[str, Any]], output_path: str):
    """
    Save list of explanation dictionaries to formatted JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2)
