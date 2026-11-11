"""
Phase 3: Experimental Validation of Context-Aware XAI
Experiment 6: Semantic and Scientific Audit of All 333 Generated Anomaly Explanations
Experiment ID: EXP-PHASE3-XAI-06

Role: Member 4 - Explainable AI, Context-Aware Analysis, and Technical Validation
Objective: Independently audit all 333 candidate-anomaly explanations for:
  - Audit A: Record Completeness and Integrity
  - Audit B: SHAP Numerical Consistency & Feature Selection
  - Audit C: Four-Model Consensus Consistency & Terminology
  - Audit D: Contextual Reference Validity & Reproducibility
  - Audit E: Scientific & Domain Validity
  - Audit F: Explanation Quality Scoring Rubric
"""

from pathlib import Path
import sys
import os
import json
import logging
import re
import numpy as np
import pandas as pd

# Setup paths
base_dir = Path(__file__).resolve().parent.parent.parent
src_dir = ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir) if ((base_dir / "Anomaly-Detection-main" if (base_dir / "Anomaly-Detection-main").exists() else base_dir)).exists() else base_dir)
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from context_analysis import load_data_splits, extract_historical_context_cohort, evaluate_contextual_metrics

FEATURE_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "lag_1",
    "lag_4",
    "rolling_mean_4",
    "rolling_std_4",
    "consumption_change",
]


def run_semantic_audit():
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
    logger = logging.getLogger("EXP-PHASE3-XAI-06")

    logger.info("=" * 76)
    logger.info("MEMBER 4 - PHASE 3: SEMANTIC & SCIENTIFIC AUDIT OF 333 EXPLANATIONS")
    logger.info("Experiment ID: EXP-PHASE3-XAI-06")
    logger.info("=" * 76)

    # 1. Load Artifacts
    logger.info("\n[Step 1] Loading existing project artifacts (Read-Only Mode)...")
    explanations_path = src_dir / "outputs" / "explanations.json"
    contextual_csv_path = src_dir / "outputs" / "contextual_anomalies.csv"
    features_csv_path = src_dir / "outputs" / "household_15min_features.csv"
    shap_ref_path = base_dir / "experiments" / "phase3_shap_stability" / "shap_values_trees_200.csv"

    stat_res_path = src_dir / "outputs" / "statistical_baseline_results.csv"
    if_res_path = src_dir / "outputs" / "isolation_forest_results.csv"
    lp_res_path = src_dir / "outputs" / "lstm_prediction_results.csv"
    lae_res_path = src_dir / "outputs" / "lstm_autoencoder_results.csv"

    paper_file_path = base_dir / "research_paper" / "complete_ieee_paper.md"

    # Strict Safeguard check on paper size
    paper_size = paper_file_path.stat().st_size
    logger.info(f"  Safeguard check: Master paper size = {paper_size:,} bytes (Required: exactly 122,229 bytes)")
    if paper_size != 122229:
        raise ValueError(f"Paper freeze violation! Expected 122,229 bytes, got {paper_size}")

    with open(explanations_path, "r", encoding="utf-8") as f:
        explanations = json.load(f)
    logger.info(f"  Loaded explanations.json: {len(explanations)} explanations")

    contextual_df = pd.read_csv(contextual_csv_path)
    logger.info(f"  Loaded contextual_anomalies.csv: {len(contextual_df)} rows")

    shap_df = pd.read_csv(shap_ref_path)
    logger.info(f"  Loaded shap_values_trees_200.csv: {len(shap_df)} rows")

    stat_res = pd.read_csv(stat_res_path)
    if_res = pd.read_csv(if_res_path)
    lp_res = pd.read_csv(lp_res_path)
    lae_res = pd.read_csv(lae_res_path)
    logger.info("  Loaded all 4 model detection result files.")

    train_df, test_df = load_data_splits(features_csv_path, train_fraction=0.8)
    logger.info(f"  Loaded data splits: Train = {len(train_df):,} rows, Test = {len(test_df):,} rows")

    # Map timestamps and records for lookup
    shap_df["ts_str"] = pd.to_datetime(shap_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    shap_map = shap_df.set_index("ts_str")

    contextual_df["ts_str"] = pd.to_datetime(contextual_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    contextual_map = contextual_df.set_index("ts_str")

    test_df["ts_str"] = pd.to_datetime(test_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    test_map = test_df.set_index("ts_str", drop=False)

    stat_res["ts_str"] = pd.to_datetime(stat_res["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if_res["ts_str"] = pd.to_datetime(if_res["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    lp_res["ts_str"] = pd.to_datetime(lp_res["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    lae_res["ts_str"] = pd.to_datetime(lae_res["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    stat_map = stat_res.set_index("ts_str")
    if_map = if_res.set_index("ts_str")
    lp_map = lp_res.set_index("ts_str")
    lae_map = lae_res.set_index("ts_str")

    # Audit tracking structures
    audit_records = []
    issue_examples = []

    summary_counts = {
        "total_anomalies_audited": len(explanations),
        "overall_status_counts": {"PASS": 0, "WARNING": 0, "FAIL": 0, "NOT_VERIFIABLE": 0},
        "dimension_status_counts": {
            "audit_a_record_completeness": {"PASS": 0, "WARNING": 0, "FAIL": 0, "NOT_VERIFIABLE": 0},
            "audit_b_shap_numerical": {"PASS": 0, "WARNING": 0, "FAIL": 0, "NOT_VERIFIABLE": 0},
            "audit_c_consensus_consistency": {"PASS": 0, "WARNING": 0, "FAIL": 0, "NOT_VERIFIABLE": 0},
            "audit_d_contextual_reference": {"PASS": 0, "WARNING": 0, "FAIL": 0, "NOT_VERIFIABLE": 0},
            "audit_e_scientific_validity": {"PASS": 0, "WARNING": 0, "FAIL": 0, "NOT_VERIFIABLE": 0},
        },
        "issue_category_counts": {
            "unsupported_appliance_attribution": 0,
            "unsupported_causal_claim": 0,
            "unsupported_household_behaviour_claim": 0,
            "incorrect_circuit_mapping": 0,
            "incorrect_shap_interpretation": 0,
            "unsupported_fault_diagnosis": 0,
            "other_scientific_terminology_issue": 0,
            "negative_or_negligible_shap_feature_cited": 0,
            "unverified_general_circuit_hypothesis": 0,
        },
        "consensus_counts_reconciled": {},
    }

    logger.info("\n[Step 2] Executing Multi-Dimensional Audit across all 333 candidate explanations...")

    seen_timestamps = set()

    for idx, expl in enumerate(explanations):
        ts = expl.get("timestamp")
        findings = []
        inst_issues = []

        dim_a_status = "PASS"
        dim_b_status = "PASS"
        dim_c_status = "PASS"
        dim_d_status = "PASS"
        dim_e_status = "PASS"

        # ---------------------------------------------------------------------
        # AUDIT A: Record Completeness and Integrity
        # ---------------------------------------------------------------------
        if not ts:
            dim_a_status = "FAIL"
            findings.append("Missing timestamp field.")
        else:
            try:
                parsed_dt = pd.to_datetime(ts)
                if ts in seen_timestamps:
                    dim_a_status = "FAIL"
                    findings.append("Duplicate timestamp detected.")
                seen_timestamps.add(ts)
            except Exception as e:
                dim_a_status = "FAIL"
                findings.append(f"Unparseable timestamp string: {ts}")

        required_keys = [
            "timestamp", "actual_power_kw", "consensus_level", "num_models_flagged",
            "model_flags", "model_scores", "context_evaluation", "submetering_supporting_clues",
            "top_shap_features", "human_readable_diagnostic"
        ]
        for rk in required_keys:
            if rk not in expl:
                dim_a_status = "FAIL"
                findings.append(f"Missing required top-level key '{rk}'.")

        ce = expl.get("context_evaluation", {})
        for cek in ["historical_expected_median_kw", "historical_expected_mean_kw", "absolute_deviation_kw", "contextual_percentile", "contextual_z_score"]:
            if cek not in ce or ce[cek] is None or (isinstance(ce[cek], float) and np.isnan(ce[cek])):
                dim_a_status = "FAIL"
                findings.append(f"Missing or NaN contextual key '{cek}'.")

        mf = expl.get("model_flags", {})
        for mfk in ["Statistical_Baseline", "Isolation_Forest", "LSTM_Prediction", "LSTM_Autoencoder"]:
            if mfk not in mf or not isinstance(mf[mfk], bool):
                dim_a_status = "FAIL"
                findings.append(f"Invalid model_flag key '{mfk}'.")

        ms = expl.get("model_scores", {})
        for msk in ["Statistical_Baseline_Score", "Isolation_Forest_Score", "LSTM_Prediction_Score", "LSTM_Autoencoder_Score"]:
            if msk not in ms or ms[msk] is None or np.isnan(ms[msk]):
                dim_a_status = "FAIL"
                findings.append(f"Missing or NaN model_score key '{msk}'.")

        # Compare with contextual_anomalies.csv
        if ts in contextual_map.index:
            csv_row = contextual_map.loc[ts]
            if abs(expl["actual_power_kw"] - csv_row["actual_power_kw"]) > 1e-4:
                dim_a_status = "FAIL"
                findings.append("actual_power_kw mismatch with contextual_anomalies.csv.")
            if abs(ce["historical_expected_median_kw"] - csv_row["context_expected_median_kw"]) > 1e-4:
                dim_a_status = "FAIL"
                findings.append("historical median mismatch with CSV.")
            if abs(ce["contextual_z_score"] - csv_row["context_z_score"]) > 1e-4:
                dim_a_status = "FAIL"
                findings.append("contextual z-score mismatch with CSV.")
        else:
            dim_a_status = "FAIL"
            findings.append("Timestamp not found in contextual_anomalies.csv.")

        # ---------------------------------------------------------------------
        # AUDIT B: SHAP Numerical Consistency & Feature Selection
        # ---------------------------------------------------------------------
        top_shap = expl.get("top_shap_features", [])
        if not top_shap or len(top_shap) != 3:
            dim_b_status = "FAIL"
            findings.append(f"Expected 3 top SHAP features, found {len(top_shap)}.")
        elif ts not in shap_map.index:
            dim_b_status = "NOT_VERIFIABLE"
            findings.append("Timestamp not found in reference 200-tree SHAP matrix.")
        else:
            ref_row = shap_map.loc[ts]
            prev_attr = float("inf")
            has_negative = False
            has_negligible = False

            for rank_i, item in enumerate(top_shap):
                f_name = item.get("feature")
                f_val = item.get("feature_value")
                f_attr = item.get("attribution")

                if f_name not in FEATURE_COLUMNS:
                    dim_b_status = "FAIL"
                    findings.append(f"Cited feature '{f_name}' not in 16 Isolation Forest features.")

                ref_attr = float(ref_row["shap_" + f_name])
                if abs(f_attr - ref_attr) > 1e-4:
                    dim_b_status = "FAIL"
                    findings.append(f"SHAP attribution mismatch for {f_name}: reported {f_attr}, reference {ref_attr}.")

                # Check descending attribution order
                if f_attr > prev_attr + 1e-6:
                    dim_b_status = "FAIL"
                    findings.append(f"Feature ordering violation at rank {rank_i}: {f_attr} > {prev_attr}.")
                prev_attr = f_attr

                # Check negative or negligible attribution
                if f_attr < 0.0:
                    has_negative = True
                elif f_attr < 0.05:
                    has_negligible = True

            # Check sign representation in text
            diag_text = expl.get("human_readable_diagnostic", "")
            for item in top_shap:
                f_name = item["feature"]
                f_attr = item["attribution"]
                if f_attr > 0:
                    if f"{f_name} =" in diag_text and "pushed anomaly score higher" not in diag_text:
                        dim_b_status = "FAIL"
                        findings.append(f"Positive attribution for {f_name} not described as pushing score higher.")
                else:
                    if f"{f_name} =" in diag_text and "acted toward normalcy" not in diag_text:
                        dim_b_status = "FAIL"
                        findings.append(f"Negative attribution for {f_name} not described as acting toward normalcy.")

            # Flag Warning if negative or negligible feature was cited under driving features
            if has_negative or has_negligible:
                if dim_b_status == "PASS":
                    dim_b_status = "WARNING"
                inst_issues.append("negative_or_negligible_shap_feature_cited")
                inst_issues.append("incorrect_shap_interpretation")
                summary_counts["issue_category_counts"]["negative_or_negligible_shap_feature_cited"] += 1
                summary_counts["issue_category_counts"]["incorrect_shap_interpretation"] += 1
                findings.append("Feature with negative (<0) or negligible (<0.05) attribution cited under 'Key driving features'.")
                
                # Record issue example
                neg_items = [f"{it['feature']} ({it['attribution']:+.2f})" for it in top_shap if it['attribution'] < 0.05]
                issue_examples.append({
                    "timestamp": ts,
                    "issue_category": "negative_or_negligible_shap_feature_cited",
                    "audit_dimension": "Audit B (SHAP Consistency)",
                    "flagged_status": "WARNING",
                    "exact_text_excerpt": f"Key driving features: {'; '.join(neg_items)}",
                    "audit_finding_reason": "Feature with negative or negligible SHAP attribution (<0.05) included in top-3 driving features list due to mechanical sorting.",
                })

        # Check that SHAP is not labeled as kW
        if "SHAP:" in expl.get("human_readable_diagnostic", ""):
            diag = expl["human_readable_diagnostic"]
            if re.search(r"SHAP:\s*[+-]?\d+(\.\d+)?\s*(kW|watts|W|kWh)", diag, re.IGNORECASE):
                dim_b_status = "FAIL"
                findings.append("SHAP attribution incorrectly labeled with physical power/energy units.")
                summary_counts["issue_category_counts"]["other_scientific_terminology_issue"] += 1

        # ---------------------------------------------------------------------
        # AUDIT C: Four-Model Consensus Consistency & Terminology
        # ---------------------------------------------------------------------
        num_flagged = expl.get("num_models_flagged", 0)
        calc_flagged = sum(1 for v in mf.values() if v)
        if num_flagged != calc_flagged:
            dim_c_status = "FAIL"
            findings.append(f"num_models_flagged ({num_flagged}) does not match True count in model_flags ({calc_flagged}).")

        # Verify against ground-truth detection CSVs
        if ts in stat_map.index:
            gt_stat = bool(stat_map.loc[ts, "is_anomaly"])
            gt_if = bool(if_map.loc[ts, "is_anomaly"])
            gt_lp = bool(lp_map.loc[ts, "is_anomaly"])
            gt_lae = bool(lae_map.loc[ts, "is_anomaly"])

            if mf.get("Statistical_Baseline") != gt_stat:
                dim_c_status = "FAIL"
                findings.append(f"Statistical_Baseline flag mismatch: {mf.get('Statistical_Baseline')} vs gt {gt_stat}.")
            if mf.get("Isolation_Forest") != gt_if:
                dim_c_status = "FAIL"
                findings.append(f"Isolation_Forest flag mismatch: {mf.get('Isolation_Forest')} vs gt {gt_if}.")
            if mf.get("LSTM_Prediction") != gt_lp:
                dim_c_status = "FAIL"
                findings.append(f"LSTM_Prediction flag mismatch: {mf.get('LSTM_Prediction')} vs gt {gt_lp}.")
            if mf.get("LSTM_Autoencoder") != gt_lae:
                dim_c_status = "FAIL"
                findings.append(f"LSTM_Autoencoder flag mismatch: {mf.get('LSTM_Autoencoder')} vs gt {gt_lae}.")
        else:
            dim_c_status = "NOT_VERIFIABLE"
            findings.append("Timestamp not found in model result CSVs.")

        # Check terminology
        cons_level = expl.get("consensus_level", "")
        if num_flagged == 4:
            if cons_level != "Four-Model Consensus Candidate Anomaly":
                dim_c_status = "FAIL"
                findings.append(f"Invalid 4-model consensus label: '{cons_level}'.")
        elif num_flagged == 3:
            if cons_level != "Three-Way Multi-Model Consensus":
                dim_c_status = "FAIL"
                findings.append(f"Invalid 3-model consensus label: '{cons_level}'.")
        elif num_flagged == 2:
            if cons_level != "Two-Model Agreement":
                dim_c_status = "FAIL"
                findings.append(f"Invalid 2-model consensus label: '{cons_level}'.")
        elif num_flagged == 1:
            if not cons_level.startswith("Isolated Flag ("):
                dim_c_status = "FAIL"
                findings.append(f"Invalid 1-model consensus label: '{cons_level}'.")

        # Check forbidden uncalibrated confidence words
        for bad_w in ["high-confidence", "calibrated confidence", "ground truth", "ground-truth", "confirmed fault", "confirmed defect"]:
            if bad_w in cons_level.lower() or bad_w in expl.get("human_readable_diagnostic", "").lower():
                dim_c_status = "FAIL"
                findings.append(f"Forbidden uncalibrated confidence/fault term '{bad_w}' detected.")

        # ---------------------------------------------------------------------
        # AUDIT D: Contextual Reference Validity & Reproducibility
        # ---------------------------------------------------------------------
        if ts in test_map.index:
            t_row = test_map.loc[ts]
            t_hour = int(t_row["hour"])
            t_dow = int(t_row["day_of_week"])
            t_weekend = int(t_row["is_weekend"])
            t_month = int(t_row["month"])

            # Recalculate context cohort from training data
            cohort = extract_historical_context_cohort(
                train_df,
                target_hour=t_hour,
                target_day_of_week=t_dow,
                target_is_weekend=t_weekend,
                target_month=t_month,
                hour_window=1,
                match_season=True,
            )

            # Assert training only (strict chronological barrier: train_df max ts < test_df min ts)
            max_train_dt = pd.to_datetime(train_df["timestamp"]).max()
            cand_dt = pd.to_datetime(ts)
            if cand_dt <= max_train_dt:
                dim_d_status = "FAIL"
                findings.append("Candidate timestamp falls inside training period (leakage check failed).")

            # Check cohort size match
            cohort_size_reported = int(contextual_map.loc[ts, "context_cohort_size"])
            if len(cohort) != cohort_size_reported:
                dim_d_status = "FAIL"
                findings.append(f"Cohort size mismatch: recalculated {len(cohort)} vs reported {cohort_size_reported}.")

            # Recalculate metrics
            recalc = evaluate_contextual_metrics(train_df, t_row, hour_window=1, match_season=True)
            if abs(ce["historical_expected_median_kw"] - recalc["context_expected_median_kw"]) > 1e-4:
                dim_d_status = "FAIL"
                findings.append("Historical median mismatch with recalculation.")
            if abs(ce["historical_expected_mean_kw"] - recalc["context_expected_mean_kw"]) > 1e-4:
                dim_d_status = "FAIL"
                findings.append("Historical mean mismatch with recalculation.")
            if abs(ce["absolute_deviation_kw"] - recalc["context_abs_deviation_kw"]) > 1e-4:
                dim_d_status = "FAIL"
                findings.append("Absolute deviation mismatch with recalculation.")
            if abs(ce["contextual_z_score"] - recalc["context_z_score"]) > 1e-4:
                dim_d_status = "FAIL"
                findings.append("Contextual z-score mismatch with recalculation.")
            if abs(ce["contextual_percentile"] - recalc["context_percentile"]) > 1e-4:
                dim_d_status = "FAIL"
                findings.append("Contextual percentile mismatch with recalculation.")
        else:
            dim_d_status = "NOT_VERIFIABLE"
            findings.append("Timestamp not found in test_df.")

        # ---------------------------------------------------------------------
        # AUDIT E: Scientific & Domain Validity
        # ---------------------------------------------------------------------
        diag_full = expl.get("human_readable_diagnostic", "")
        clues_full = " ".join(expl.get("submetering_supporting_clues", []))

        # 1. Check unsupported appliance names
        unsupported_appliances = [
            "microwave", "dishwasher", "tumble dryer", "tumble-drier", "washing machine",
            "refrigerator", "freezer", "fridge", "compressor", "toaster", "kettle", "oven"
        ]
        for app in unsupported_appliances:
            if re.search(r"\b" + re.escape(app) + r"\b", diag_full, re.IGNORECASE):
                dim_e_status = "FAIL"
                inst_issues.append("unsupported_appliance_attribution")
                summary_counts["issue_category_counts"]["unsupported_appliance_attribution"] += 1
                findings.append(f"Unsupported specific appliance attribution: '{app}'.")
                issue_examples.append({
                    "timestamp": ts,
                    "issue_category": "unsupported_appliance_attribution",
                    "audit_dimension": "Audit E (Scientific Validity)",
                    "flagged_status": "FAIL",
                    "exact_text_excerpt": app,
                    "audit_finding_reason": f"Specific unmeasured appliance '{app}' cited without telemetry proof.",
                })

        # 2. Check unsupported causal or fault claims
        causal_fault_terms = ["caused by", "malfunction", "equipment fault", "short circuit", "damaged appliance", "defect"]
        for cft in causal_fault_terms:
            if re.search(r"\b" + re.escape(cft) + r"\b", diag_full, re.IGNORECASE):
                dim_e_status = "FAIL"
                inst_issues.append("unsupported_causal_claim")
                summary_counts["issue_category_counts"]["unsupported_causal_claim"] += 1
                findings.append(f"Unsupported causal/fault claim: '{cft}'.")
                issue_examples.append({
                    "timestamp": ts,
                    "issue_category": "unsupported_causal_claim",
                    "audit_dimension": "Audit E (Scientific Validity)",
                    "flagged_status": "FAIL",
                    "exact_text_excerpt": cft,
                    "audit_finding_reason": f"Unverified causal/fault claim '{cft}' asserted without physical ground-truth inspection.",
                })

        # 3. Check household routine claims
        routine_terms = ["occupancy routine", "occupant behavior", "user habit", "sleeping routine"]
        for rt in routine_terms:
            if re.search(r"\b" + re.escape(rt) + r"\b", diag_full, re.IGNORECASE):
                dim_e_status = "FAIL"
                inst_issues.append("unsupported_household_behaviour_claim")
                summary_counts["issue_category_counts"]["unsupported_household_behaviour_claim"] += 1
                findings.append(f"Unsupported household routine claim: '{rt}'.")

        # 4. Check circuit mapping correctness
        if "Sub-metering 1" in clues_full and "kitchen" not in clues_full.lower():
            dim_e_status = "FAIL"
            inst_issues.append("incorrect_circuit_mapping")
            summary_counts["issue_category_counts"]["incorrect_circuit_mapping"] += 1
            findings.append("Sub-metering 1 not correctly mapped to kitchen circuit.")
        if "Sub-metering 2" in clues_full and "laundry" not in clues_full.lower():
            dim_e_status = "FAIL"
            inst_issues.append("incorrect_circuit_mapping")
            summary_counts["issue_category_counts"]["incorrect_circuit_mapping"] += 1
            findings.append("Sub-metering 2 not correctly mapped to laundry-room circuit.")
        if "Sub-metering 3" in clues_full and ("water heater" not in clues_full.lower() and "air-conditioning" not in clues_full.lower()):
            dim_e_status = "FAIL"
            inst_issues.append("incorrect_circuit_mapping")
            summary_counts["issue_category_counts"]["incorrect_circuit_mapping"] += 1
            findings.append("Sub-metering 3 not correctly mapped to water heater / HVAC circuit.")

        # 5. Check unmetered general circuits speculative hypothesis
        if "unmetered general circuits, lighting, or wall sockets" in clues_full:
            # This is framed cautiously ('suggesting that...'), but names unmetered end-uses (lighting, wall sockets)
            # that have no dedicated circuit sub-meters. Classified as WARNING for scientific rigor.
            if dim_e_status == "PASS":
                dim_e_status = "WARNING"
            inst_issues.append("unverified_general_circuit_hypothesis")
            summary_counts["issue_category_counts"]["unverified_general_circuit_hypothesis"] += 1
            findings.append("Tentative unmetered circuit hypothesis ('lighting, or wall sockets') generated without dedicated sub-meter verification.")
            issue_examples.append({
                "timestamp": ts,
                "issue_category": "unverified_general_circuit_hypothesis",
                "audit_dimension": "Audit E (Scientific Validity)",
                "flagged_status": "WARNING",
                "exact_text_excerpt": "suggesting that the primary load deviation derived from unmetered general circuits, lighting, or wall sockets.",
                "audit_finding_reason": "End-use hypothesis (lighting/wall sockets) lacks sub-meter validation; represents unmeasured residual load.",
            })

        # ---------------------------------------------------------------------
        # AUDIT F: Explanation Quality Scoring Synthesis
        # ---------------------------------------------------------------------
        statuses = [dim_a_status, dim_b_status, dim_c_status, dim_d_status, dim_e_status]
        if "FAIL" in statuses:
            overall_status = "FAIL"
        elif "WARNING" in statuses:
            overall_status = "WARNING"
        elif "NOT_VERIFIABLE" in statuses:
            overall_status = "NOT_VERIFIABLE"
        else:
            overall_status = "PASS"

        # Update summary counts
        summary_counts["overall_status_counts"][overall_status] += 1
        summary_counts["dimension_status_counts"]["audit_a_record_completeness"][dim_a_status] += 1
        summary_counts["dimension_status_counts"]["audit_b_shap_numerical"][dim_b_status] += 1
        summary_counts["dimension_status_counts"]["audit_c_consensus_consistency"][dim_c_status] += 1
        summary_counts["dimension_status_counts"]["audit_d_contextual_reference"][dim_d_status] += 1
        summary_counts["dimension_status_counts"]["audit_e_scientific_validity"][dim_e_status] += 1

        finding_str = "; ".join(findings) if findings else "All audit checks passed cleanly."
        issues_str = "; ".join(sorted(list(set(inst_issues)))) if inst_issues else "None"

        audit_records.append({
            "timestamp": ts,
            "actual_power_kw": float(expl.get("actual_power_kw", 0.0)),
            "models_flagged_count": num_flagged,
            "consensus_level": cons_level,
            "audit_a_completeness_status": dim_a_status,
            "audit_b_shap_status": dim_b_status,
            "audit_c_consensus_status": dim_c_status,
            "audit_d_context_status": dim_d_status,
            "audit_e_scientific_status": dim_e_status,
            "overall_status": overall_status,
            "issue_categories": issues_str,
            "concise_findings": finding_str,
        })

    # Summary consensus breakdown
    c_counts = {}
    for r in audit_records:
        k = r["models_flagged_count"]
        c_counts[k] = c_counts.get(k, 0) + 1
    summary_counts["consensus_counts_reconciled"] = {f"{k}_model_consensus": v for k, v in sorted(c_counts.items())}

    # Save deliverables
    logger.info("\n[Step 3] Exporting deliverables to experiments/phase3_explanation_semantic_audit/...")

    audit_df = pd.DataFrame(audit_records)
    audit_csv_path = exp_dir / "explanation_semantic_audit.csv"
    audit_df.to_csv(audit_csv_path, index=False)
    logger.info(f"  Saved explanation_semantic_audit.csv ({len(audit_df)} rows, {len(audit_df.columns)} cols)")

    issue_df = pd.DataFrame(issue_examples)
    issue_csv_path = exp_dir / "explanation_issue_examples.csv"
    issue_df.to_csv(issue_csv_path, index=False)
    logger.info(f"  Saved explanation_issue_examples.csv ({len(issue_df)} flagged items across {len(issue_df['timestamp'].unique()) if len(issue_df) else 0} unique timestamps)")

    summary_json_path = exp_dir / "explanation_quality_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_counts, f, indent=2)
    logger.info(f"  Saved explanation_quality_summary.json")

    # Logging results summary
    logger.info("\n" + "=" * 76)
    logger.info("AUDIT SUMMARY RESULTS (N = 333 CANDIDATE EXPLANATIONS):")
    logger.info("=" * 76)
    logger.info(f"  Overall Status:")
    logger.info(f"    PASS:           {summary_counts['overall_status_counts']['PASS']} ({summary_counts['overall_status_counts']['PASS']/333*100:.2f}%)")
    logger.info(f"    WARNING:        {summary_counts['overall_status_counts']['WARNING']} ({summary_counts['overall_status_counts']['WARNING']/333*100:.2f}%)")
    logger.info(f"    FAIL:           {summary_counts['overall_status_counts']['FAIL']} ({summary_counts['overall_status_counts']['FAIL']/333*100:.2f}%)")
    logger.info(f"    NOT_VERIFIABLE: {summary_counts['overall_status_counts']['NOT_VERIFIABLE']} (0.00%)")
    logger.info("\n  Dimension-Wise Breakdown:")
    for dim_k, counts in summary_counts["dimension_status_counts"].items():
        logger.info(f"    {dim_k:<32}: PASS={counts['PASS']:<3} WARNING={counts['WARNING']:<3} FAIL={counts['FAIL']:<3}")
    logger.info("\n  Consensus Counts Reconciled:")
    for ck, cv in summary_counts["consensus_counts_reconciled"].items():
        logger.info(f"    {ck}: {cv}")
    logger.info("\n  Issue Categories Detected:")
    for ik, iv in summary_counts["issue_category_counts"].items():
        logger.info(f"    {ik:<44}: {iv}")
    logger.info("=" * 76)

    return audit_df, summary_counts, issue_df


if __name__ == "__main__":
    run_semantic_audit()

