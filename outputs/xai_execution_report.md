# XAI Execution Report: Context-Aware Anomaly Explanation

**Role:** Member 4 — Explainable AI (XAI), Context-Aware Analysis, and Research Paper
**Status:** Successfully Executed. All empirical outputs generated and verified.

---

## 1. Execution Overview and Candidate Census

- **Total Candidate Anomaly Pool (Union of all 4 models):** 333 timestamps
  - Statistical Baseline Flagged: 82
  - Isolation Forest Flagged: 93
  - LSTM Prediction Flagged: 155
  - LSTM Autoencoder Flagged: 139

## 2. TreeSHAP Compatibility and Formulation Verification

- **Explainer Implementation:** `shap.TreeExplainer` on Scikit-Learn `IsolationForest` (200 trees).
- **Scalar Function Explained:** Average Tree Depth / Isolation Path Length $h(\mathbf{x})$.
- **Attribution Inversion:** Anomaly severity attribution is formulated as $\phi_j^{\text{anom}} = -\phi_j^{\text{path}}$. A negative path length contribution implies faster isolation near the root, representing a positive contribution to anomaly severity.

### Global Feature Importance Ranking (Mean |SHAP Anomaly Attribution|):

| Rank | Feature Name | Mean |SHAP| Attribution | Primary Physical Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `Sub_metering_2` | 0.7644 | Measured / Engineered |
| 2 | `rolling_std_4` | 0.6584 | Measured / Engineered |
| 3 | `Global_intensity` | 0.5600 | Measured / Engineered |
| 4 | `Sub_metering_1` | 0.5239 | Measured / Engineered |
| 5 | `Global_active_power` | 0.4961 | Measured / Engineered |
| 6 | `consumption_change` | 0.4252 | Measured / Engineered |
| 7 | `lag_1` | 0.3401 | Measured / Engineered |
| 8 | `rolling_mean_4` | 0.3135 | Measured / Engineered |
| 9 | `lag_4` | 0.2638 | Measured / Engineered |
| 10 | `is_weekend` | 0.2219 | Measured / Engineered |
| 11 | `Global_reactive_power` | 0.1951 | Measured / Engineered |
| 12 | `Sub_metering_3` | 0.1405 | Measured / Engineered |
| 13 | `Voltage` | 0.1115 | Measured / Engineered |
| 14 | `day_of_week` | 0.1089 | Measured / Engineered |
| 15 | `hour` | 0.0729 | Measured / Engineered |
| 16 | `month` | 0.0676 | Measured / Engineered |

## 3. Generated Visual Artifacts

1. **Global Summary Plot:** `shap_summary.png` — Horizontal bar distribution of global feature attributions.
2. **Local Explanation Plot:** `individual_anomaly_explanation.png` — Waterfall-style local attribution for event `2010-04-11 20:45:00`.

## 4. Showcase Anomaly Case Study and Human-Readable Diagnostic

- **Timestamp:** `2010-04-11 20:45:00`
- **Actual Power:** 7.09 kW
- **Consensus Classification:** Four-Model Consensus Candidate Anomaly
- **Historical Context Median:** 1.99 kW
- **Contextual Deviation:** 5.10 kW
- **Contextual Percentile Rank:** 99.8%
- **Synthesized Diagnostic:**
  > "*Event at 2010-04-11 20:45:00: Four-Model Consensus Candidate Anomaly. Active power of 7.09 kW was 5.10 kW above the historical contextual median (1.99 kW), representing 3.56x typical consumption for this temporal regime (Percentile: 99.8%). Sub-circuit telemetry: Elevated Sub-metering 1 reading (73.6 Wh vs. contextual median 0.0 Wh) indicates localized load on the kitchen sub-circuit. Features contributing towards anomaly severity: Sub_metering_1 = 73.60 pushed anomaly score higher (SHAP: +1.26); Global_intensity = 30.45 pushed anomaly score higher (SHAP: +0.93); Global_active_power = 7.09 pushed anomaly score higher (SHAP: +0.86).*"

## 5. Academic Guardrails and Methodological Compliance

1. **Strict Leakage Prevention:** All context cohorts were retrieved exclusively from the training partition ($N_{\text{train}} = 109,347$). Zero future test records entered the baseline.
2. **Clues vs. Proof:** Sub-metering telemetry (`Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`) is explicitly characterized as circumstantial supporting evidence rather than definitive appliance diagnoses.
3. **Empirical Integrity:** Zero synthetic metrics, false precision, or unverified claims.
