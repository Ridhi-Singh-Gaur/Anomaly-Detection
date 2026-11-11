# Member 4 — Phase 3: Experimental Validation of Context-Aware XAI
## Experiment 6: Semantic and Scientific Audit of All 333 Generated Anomaly Explanations Report

**Author:** Member 4 — Explainable AI (XAI), Context-Aware Analysis, and Technical Validation  
**Experiment Identifier:** `EXP-PHASE3-XAI-06`  
**Execution Date:** 2026-11-07  
**Status:** Audit Fully Completed, All 333 Explanations Evaluated, 45/45 Unit Tests Passing  
**Integrity Safeguard:** Research paper master draft remains **strictly frozen** ([`complete_ieee_paper.md`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/research_paper/complete_ieee_paper.md) verified at exactly **122,229 bytes**). Production pipelines, fitted model weights, and baseline outputs remain unmodified.

---

## 1. Executive Summary & Audit Mandate

In Phase 2 and early Phase 3 contributions, Member 4 implemented a unified Context-Aware Explainable AI (XAI) pipeline that combines:
1. Multi-model consensus voting across four distinct detectors (Statistical Baseline, Isolation Forest, LSTM Predictor, LSTM Autoencoder).
2. Dynamic historical twin matching against training-only operational cohorts.
3. 200-tree game-theoretic TreeSHAP feature attributions explaining inverted average path length.
4. Deterministic natural-language diagnostic explanation generation.

The purpose of **Experiment 6 (`EXP-PHASE3-XAI-06`)** is to conduct an independent, read-only scientific and semantic audit of all **333 generated candidate anomaly explanations** ([`explanations.json`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/outputs/explanations.json)). This audit does not manipulate or rewrite existing outputs, but independently scrutinizes every explanation across five foundational dimensions:
- **Audit A:** Record Completeness and Structural Integrity
- **Audit B:** SHAP Numerical Consistency and Feature Selection
- **Audit C:** Four-Model Consensus Consistency and Terminology
- **Audit D:** Contextual Reference Validity and Mathematical Reproducibility
- **Audit E:** Scientific and Domain Validity (Appliance/Causal Restraint)
- **Audit F:** Multi-Dimensional Quality Scoring Rubric

### Key Audit Findings
1. **Zero Structural Failures (100% Audit Coverage):** All 333 candidate anomalies possess valid, unique, and parseable timestamps. No records are missing or duplicated, and all contextual metrics reconcile with [`contextual_anomalies.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/outputs/contextual_anomalies.csv).
2. **Zero Temporal Leakage:** Historical context cohorts for all 333 anomalies are drawn exclusively from the training partition ($N_{\text{train}} = 109,347$ observations, ending 2009-12-11). All 333 candidate anomalies occur during the test period in 2010. Contextual cohort sizes range from $888$ to $2,608$ observations (mean $1,805.6$), requiring zero fallbacks.
3. **Consensus Counts Exactly Reconciled:** The model flags in [`explanations.json`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/outputs/explanations.json) match the ground-truth prediction outputs from the four models with $100\%$ precision:
   - Exactly 1 model: **242**
   - Exactly 2 models: **50**
   - Exactly 3 models: **37**
   - Exactly 4 models: **4**
   - Total Unique Candidates: **333**
4. **Zero Unsupported Specific Appliance or Causal Claims:** Phase 2 corrections successfully eliminated all ungrounded appliance-specific attributions (microwave, dishwasher, dryer, refrigerator compressor) and causal defect language ("malfunction", "damaged equipment").
5. **Identification of 78 Transparent Warnings across 73 Explanations:**
   - **Audit B Warnings (5 instances):** In 5 candidate anomalies, mechanical sorting of the top-3 SHAP features included features with negative attributions ($\phi < 0$) or negligible magnitudes ($|\phi| < 0.05$) under the heading *"Key driving features"*, because only 1 or 2 features pushed anomaly severity positively.
   - **Audit E Warnings (73 instances):** In 73 candidate anomalies where sub-metering channels (Sub1, Sub2, Sub3) were low or near typical levels, the explanation generated a cautious hypothesis: *"suggesting that the primary load deviation derived from unmetered general circuits, lighting, or wall sockets."* While framed as a suggestion, naming "lighting, or wall sockets" is an unmeasured end-use hypothesis that cannot be verified from the data.
6. **Overall Quality Scoring Distribution:**
   - **PASS:** **260 explanations ($78.08\%$)**
   - **WARNING:** **73 explanations ($21.92\%$)**
   - **FAIL:** **0 explanations ($0.00\%$)**
   - **NOT VERIFIABLE:** **0 explanations ($0.00\%$)**

---

## 2. Multi-Dimensional Quality Scoring Rubric

Each candidate anomaly explanation is evaluated across five distinct dimensions, yielding dimension-specific statuses and a unified overall quality rating.

```
       [Audit A: Record Completeness]
       [Audit B: SHAP Numerical Consistency]
Candidate ───► [Audit C: Model-Consensus Consistency] ───► Overall Status:
Explanation    [Audit D: Contextual Reference Validity]        PASS / WARNING / FAIL
       [Audit E: Scientific / Domain Validity]
```

### Rubric Definitions

| Status | Definition | Operational Criteria |
| :--- | :--- | :--- |
| **PASS** | Meets all scientific and numerical standards. | Full data availability, zero numerical discrepancy ($< 10^{-4}$), proper terminology, no ungrounded claims, and non-negligible positive driving features. |
| **WARNING** | Methodologically sound, but exhibits a documented heuristic or interpretive limitation. | Valid numbers and formatting, but contains heuristic edge cases (e.g., negative/negligible SHAP feature cited under driving features) or unverified end-use hypotheses. |
| **FAIL** | Serious defect, discrepancy, or scientific violation. | Missing record, numerical mismatch ($> 10^{-4}$), invalid feature column, ungrounded specific appliance claim, causal fault assertion, or temporal leakage. |
| **NOT VERIFIABLE** | Insufficient baseline or reference data to evaluate. | Ground-truth model output, SHAP reference matrix, or test feature row is missing. |

### Synthesis Rule
$$\text{Overall Status} = \begin{cases} \text{FAIL}, & \text{if any dimension is FAIL} \\ \text{WARNING}, & \text{else if any dimension is WARNING} \\ \text{NOT VERIFIABLE}, & \text{else if any dimension is NOT VERIFIABLE} \\ \text{PASS}, & \text{if all five dimensions are PASS} \end{cases}$$

---

## 3. Quantitative Audit Findings

### 3.1 Overall Quality Distribution ($N = 333$ Explanations)

| Overall Status | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **PASS** | **260** | **$78.08\%$** | Flawless record integrity, exact SHAP numbers, proper consensus, valid context, and strict domain restraint. |
| **WARNING** | **73** | **$21.92\%$** | Methodologically valid explanations with documented heuristic warnings (68 with unmetered hypothesis; 5 with both unmetered hypothesis and negative/negligible SHAP feature). |
| **FAIL** | **0** | **$0.00\%$** | Zero numerical mismatches, zero temporal leakages, and zero ungrounded appliance defect claims. |
| **NOT VERIFIABLE** | **0** | **$0.00\%$** | Complete audit coverage across 100% of candidate instances. |
| **Total Audited** | **333** | **$100.00\%$** | Exhaustive census of all multi-model candidate anomalies. |

### 3.2 Dimension-by-Dimension Audit Statistics

| Audit Dimension | PASS | WARNING | FAIL | NOT VERIFIABLE | Key Findings |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Audit A: Record Completeness** | **333** ($100\%$) | 0 ($0\%$) | 0 ($0\%$) | 0 ($0\%$) | Exactly 333 unique parseable timestamps; zero NaNs; full schema conformity; exact CSV alignment. |
| **Audit B: SHAP Numerical** | **328** ($98.5\%$) | **5** ($1.5\%$) | 0 ($0\%$) | 0 ($0\%$) | Zero numerical mismatches against 200-tree reference matrix; 5 instances cite $\phi < 0$ or $|\phi| < 0.05$ under driving features. |
| **Audit C: Consensus Consistency** | **333** ($100\%$) | 0 ($0\%$) | 0 ($0\%$) | 0 ($0\%$) | Exact reconciliation with 4-model detector predictions; strict adherence to established consensus labels. |
| **Audit D: Contextual Reference** | **333** ($100\%$) | 0 ($0\%$) | 0 ($0\%$) | 0 ($0\%$) | 100% training-only reference cohorts; zero temporal leakage; exact numerical recalculation match. |
| **Audit E: Scientific Validity** | **260** ($78.1\%$) | **73** ($21.9\%$) | 0 ($0\%$) | 0 ($0\%$) | Zero unsupported appliance/causal claims; 73 instances generate cautious unmetered circuit hypotheses. |

### 3.3 Four-Model Consensus Count Reconciliation

| Consensus Tier | Flagged Model Count | Verified Audit Count | Expected Baseline | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Isolated Flag** | 1 model | **242** | 242 | **EXACT MATCH** |
| - *LSTM Autoencoder* | 1 model | 110 | 110 | EXACT MATCH |
| - *LSTM Predictor* | 1 model | 96 | 96 | EXACT MATCH |
| - *Isolation Forest* | 1 model | 18 | 18 | EXACT MATCH |
| - *Statistical Baseline* | 1 model | 18 | 18 | EXACT MATCH |
| **Two-Model Agreement** | 2 models | **50** | 50 | **EXACT MATCH** |
| **Three-Way Multi-Model Consensus** | 3 models | **37** | 37 | **EXACT MATCH** |
| **Four-Model Consensus Candidate Anomaly** | 4 models | **4** | 4 | **EXACT MATCH** |
| **Total Unique Candidate Anomalies** | $\mathbf{\ge 1}$ **model** | **333** | **333** | **EXACT MATCH** |

### 3.4 Issue Category Breakdown

| Issue Category Code | Frequency | Severity | Root Cause & Scientific Impact |
| :--- | :---: | :---: | :--- |
| `unsupported_appliance_attribution` | **0** | FAIL | Successfully eradicated in Phase 2; no microwaves, dishwashers, or dryers named. |
| `unsupported_causal_claim` | **0** | FAIL | Zero assertions of equipment malfunctions, physical faults, or defect causality. |
| `unsupported_household_behaviour_claim` | **0** | FAIL | Zero assertions regarding occupant presence, sleeping habits, or routine schedules. |
| `incorrect_circuit_mapping` | **0** | FAIL | 100% adherence to Sub1=kitchen, Sub2=laundry, Sub3=water heater/HVAC. |
| `unsupported_fault_diagnosis` | **0** | FAIL | Zero claims asserting that telemetry anomalies represent confirmed equipment failures. |
| `other_scientific_terminology_issue` | **0** | FAIL | Zero instances of SHAP values labeled in kW; consensus labels strictly non-calibrated. |
| `negative_or_negligible_shap_feature_cited` | **5** | WARNING | Mechanical truncation of top-3 features included negative or negligible features ($|\phi| < 0.05$). |
| `incorrect_shap_interpretation` | **5** | WARNING | Listing normalizing features ($\phi < 0$) under the text header "Key driving features". |
| `unverified_general_circuit_hypothesis` | **73** | WARNING | Inferring "lighting, or wall sockets" when sub-metering is low; represents unmeasured residual load. |

---

## 4. Deep-Dive Dimension-by-Dimension Findings

### 4.1 Audit A: Record Completeness and Structural Integrity
- **Timestamp Validation:** All 333 records have timestamps formatted strictly as `YYYY-MM-DD HH:MM:SS`. Every timestamp is unique; zero duplicates exist.
- **Key Population:** All 10 required top-level keys (`timestamp`, `actual_power_kw`, `consensus_level`, `num_models_flagged`, `model_flags`, `model_scores`, `context_evaluation`, `submetering_supporting_clues`, `top_shap_features`, `human_readable_diagnostic`) are populated in $100\%$ of records.
- **Cross-File Reconciliation:** Every explanation record was matched against [`contextual_anomalies.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/outputs/contextual_anomalies.csv). Actual power readings, contextual expected medians, absolute deviations, and contextual z-scores match within $< 10^{-4}$ tolerance.

### 4.2 Audit B: SHAP Numerical Consistency & Feature Selection
- **Feature Name Validation:** All features cited in `top_shap_features` belong to the 16 features used by the trained Isolation Forest model.
- **Numerical Verification:** Attributions were cross-checked against the reference 200-tree SHAP matrix ([`shap_values_trees_200.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_stability/shap_values_trees_200.csv)). Zero numerical discrepancies were detected across all $333 \times 3 = 999$ attribution pairs ($|attr_{\text{reported}} - attr_{\text{reference}}| < 10^{-6}$).
- **Feature Ordering:** For all 333 explanations, features are sorted in descending order of attribution magnitude ($attr_0 \ge attr_1 \ge attr_2$).
- **The Top-K Heuristic Limitation (5 Warnings):**
  - In `context_aware_xai.py`, the pipeline mechanically extracts the top 3 features by sorted attribution (`feat_order[:3]`).
  - In 5 candidate anomaly instances, the Isolation Forest isolated the instance primarily on a single feature (`rolling_std_4`), leaving the remaining 15 features with near-zero or slightly negative attributions.
  - As a result, rank 2 and/or rank 3 contained negative attributions ($\phi < 0$) or negligible magnitudes ($|\phi| < 0.05$).
  - While the explanation generator correctly appended `"acted toward normalcy (SHAP: -0.02)"`, placing this statement under the sentence header `"Key driving features:"` represents an interpretive heuristic limitation.

### 4.3 Audit C: Four-Model Consensus Consistency & Terminology
- **Flag Reconciliation:** Each model flag in `model_flags` was verified against the raw anomaly prediction arrays from:
  - [`statistical_baseline_results.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/outputs/statistical_baseline_results.csv)
  - [`isolation_forest_results.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/outputs/isolation_forest_results.csv)
  - [`lstm_prediction_results.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/outputs/lstm_prediction_results.csv)
  - [`lstm_autoencoder_results.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/outputs/lstm_autoencoder_results.csv)
  Zero mismatches were found. The sum of True flags equals `num_models_flagged` in all 333 records.
- **Consensus Terminology:** The 4 consensus levels strictly adhere to non-calibrated designations:
  - 4 models flagged: `"Four-Model Consensus Candidate Anomaly"` ($N = 4$)
  - 3 models flagged: `"Three-Way Multi-Model Consensus"` ($N = 37$)
  - 2 models flagged: `"Two-Model Agreement"` ($N = 50$)
  - 1 model flagged: `"Isolated Flag (<Model_Name>)"` ($N = 242$)
- **Calibrated Confidence Avoidance:** Zero occurrences of `"high-confidence"`, `"ground truth"`, or `"probability"` were found.

### 4.4 Audit D: Contextual Reference Validity & Reproducibility
- **Chronological Boundary Integrity:**
  - The training dataset ends at `2009-12-11 23:45:00` ($N = 109,347$ observations).
  - The earliest candidate anomaly occurs at `2010-01-13 18:30:00`, and the latest at `2010-11-26 12:45:00`.
  - Temporal leakage is strictly $0.0\%$. No future or test-period observation enters any historical cohort.
- **Cohort Extraction and Size Sufficiency:**
  - Every historical cohort was independently reconstructed using the identical matching rules ($\text{hour} \pm 1$ modular 24, identical $\text{is\_weekend}$, $\text{month} \pm 1$ modular 12).
  - Cohort sizes range from $888$ to $2,608$ training observations (mean $1,805.6$). Because all cohort sizes substantially exceed the 50-sample threshold, zero fallbacks were triggered.
- **Mathematical Reproducibility:**
  - Recalculated medians, means, absolute deviations, z-scores, and percentiles match reported values to within floating-point precision ($< 10^{-6}$).

### 4.5 Audit E: Scientific and Domain Validity
- **Circuit Mapping:** Circuit descriptions strictly follow the UCI dataset documentation:
  - `Sub_metering_1`: Kitchen sub-circuit (microwave, dishwasher, hot-plates).
  - `Sub_metering_2`: Laundry room sub-circuit (washing machine, tumble-drier, refrigerator, light).
  - `Sub_metering_3`: Electric water heater and air conditioning.
- **Elimination of Unsupported Appliance Claims:** Phase 2 refactoring completely removed all specific appliance attributions. Telemetry is described at the circuit level (e.g., *"indicates localized load on the kitchen sub-circuit"*).
- **The Unmetered Load Hypothesis (73 Warnings):**
  - When all three sub-metering channels are near zero or typical levels, the explanation outputs:
    `Sub-metering channels (...) remained relatively low or near typical levels, suggesting that the primary load deviation derived from unmetered general circuits, lighting, or wall sockets.`
  - The phrasing uses tentative language (*"suggesting that..."*), so it does not assert this as fact.
  - However, because lighting and wall sockets are not directly sub-metered in the UCI telemetry, inferring that the residual load derived specifically from lighting or wall sockets represents an unverified end-use hypothesis. In a rigorous scientific audit, these 73 explanations are flagged as `WARNING` to maintain complete scientific transparency.

---

## 5. Representative Successful Explanations (PASS)

### Example 1: Four-Model Consensus Candidate Anomaly
- **Timestamp:** `2010-02-13 12:30:00`
- **Actual Power:** $7.01\text{ kW}$ (Percentile: $99.9\%$, Z-score: $4.29$)
- **Contextual Median:** $1.62\text{ kW}$ (Deviation: $+5.39\text{ kW}$, Ratio: $4.33\times$)
- **Models Flagged (4/4):** Statistical Baseline ($1.72$), Isolation Forest ($+0.064$), LSTM Predictor ($2.08$), LSTM Autoencoder ($28.05$)
- **Sub-circuit Telemetry:** Elevated Sub1 ($17.8\text{ Wh}$ vs. $0.0\text{ Wh}$), Elevated Sub2 ($63.5\text{ Wh}$ vs. $0.0\text{ Wh}$)
- **Top SHAP Attributions:** `Sub_metering_2` ($+1.58$), `Global_intensity` ($+0.85$), `Global_active_power` ($+0.84$)
- **Synthesized Explanation:**
  > *"Event at 2010-02-13 12:30:00: Four-Model Consensus Candidate Anomaly. Active power of 7.01 kW was 5.39 kW above the historical contextual median (1.62 kW), representing 4.33x typical consumption for this temporal regime (Percentile: 99.9%). Sub-circuit telemetry: Elevated Sub-metering 1 reading (17.8 Wh vs. contextual median 0.0 Wh) indicates localized load on the kitchen sub-circuit. Elevated Sub-metering 2 reading (63.5 Wh vs. contextual median 0.0 Wh) indicates localized load on the laundry-room sub-circuit. Key driving features: Sub_metering_2 = 63.47 pushed anomaly score higher (SHAP: +1.58); Global_intensity = 29.84 pushed anomaly score higher (SHAP: +0.85); Global_active_power = 7.01 pushed anomaly score higher (SHAP: +0.84)."*
- **Audit Verdict:** **PASS (All 5 Dimensions)**. Flawless reconciliation, strong multi-circuit telemetry support, high positive SHAP values, and strict circuit-level language.

### Example 2: Kitchen Circuit Excursion (Three-Way Consensus)
- **Timestamp:** `2010-02-06 18:45:00`
- **Actual Power:** $6.40\text{ kW}$ (Percentile: $98.7\%$, Z-score: $2.65$)
- **Contextual Median:** $2.46\text{ kW}$ (Deviation: $+3.94\text{ kW}$)
- **Models Flagged (3/4):** Statistical Baseline, Isolation Forest, LSTM Predictor
- **Sub-circuit Telemetry:** Elevated Sub1 ($48.7\text{ Wh}$ vs. $0.0\text{ Wh}$)
- **Top SHAP Attributions:** `Sub_metering_1` ($+1.17$), `Global_intensity` ($+0.93$), `Global_active_power` ($+0.78$)
- **Audit Verdict:** **PASS (All 5 Dimensions)**. Kitchen circuit load directly corroborated by SHAP Rank 1 attribution on `Sub_metering_1`.

### Example 3: Acute Consumption Spike on Laundry Circuit
- **Timestamp:** `2010-02-07 14:45:00`
- **Actual Power:** $3.28\text{ kW}$ (Contextual Median: $1.64\text{ kW}$)
- **Models Flagged (1/4):** Isolated Flag (LSTM Predictor)
- **Sub-circuit Telemetry:** Elevated Sub2 ($46.6\text{ Wh}$ vs. $0.2\text{ Wh}$)
- **Top SHAP Attributions:** `Sub_metering_2` ($+2.10$), `consumption_change` ($+0.77$), `rolling_std_4` ($+0.77$)
- **Audit Verdict:** **PASS (All 5 Dimensions)**. Dynamic consumption spike corroborated by both circuit telemetry and temporal gradient features.

---

## 6. Representative Problematic Explanations (WARNINGs)

### Problematic Case 1: Negative & Negligible SHAP Features Cited
- **Timestamp:** `2010-05-19 16:30:00`
- **Consensus Level:** Isolated Flag (LSTM_Autoencoder)
- **Actual Power:** $0.40\text{ kW}$ (Percentile: $51.1\%$, Z-score: $-0.02$)
- **Contextual Median:** $0.40\text{ kW}$ (Deviation: $0.00\text{ kW}$)
- **Top SHAP Attributions:**
  - Rank 1: `rolling_std_4` = $1.06$ (Attribution: **$+0.91$**)
  - Rank 2: `rolling_mean_4` = $0.95$ (Attribution: **$-0.02$** $\rightarrow$ Negative)
  - Rank 3: `hour` = $16.00$ (Attribution: **$-0.06$** $\rightarrow$ Negative)
- **Diagnostic Text:**
  > *"Event at 2010-05-19 16:30:00: Isolated Flag (LSTM_Autoencoder). Active power of 0.40 kW was 0.00 kW above the historical contextual median (0.40 kW), representing 1.01x typical consumption for this temporal regime (Percentile: 51.1%). Sub-circuit telemetry: Sub-metering channels (Sub1: 0.0 Wh, Sub2: 0.0 Wh, Sub3: 0.7 Wh) remained relatively low or near typical levels, suggesting that the primary load deviation derived from unmetered general circuits, lighting, or wall sockets. Key driving features: rolling_std_4 = 1.06 pushed anomaly score higher (SHAP: +0.91); rolling_mean_4 = 0.95 acted toward normalcy (SHAP: -0.02); hour = 16.00 acted toward normalcy (SHAP: -0.06)."*
- **Audit Finding:**
  - **Audit B: WARNING** (`negative_or_negligible_shap_feature_cited`, `incorrect_shap_interpretation`). Features that acted toward normalcy ($\phi < 0$) were placed under the text header *"Key driving features"*.
  - **Audit E: WARNING** (`unverified_general_circuit_hypothesis`). Sub-metering is low, so the explanation tentatively speculates about lighting or wall sockets.
  - **Overall Status:** **WARNING**.

### Problematic Case 2: Near-Zero Attribution Cited as Driving Feature
- **Timestamp:** `2010-08-27 16:30:00`
- **Consensus Level:** Isolated Flag (LSTM_Autoencoder)
- **Top SHAP Attributions:** `rolling_std_4` ($+1.55$), `rolling_mean_4` ($+0.18$), `Voltage` (**$-0.0035$**)
- **Diagnostic Text Excerpt:**
  > *"Key driving features: rolling_std_4 = 1.60 pushed anomaly score higher (SHAP: +1.55); rolling_mean_4 = 1.61 pushed anomaly score higher (SHAP: +0.18); Voltage = 243.75 acted toward normalcy (SHAP: -0.00)."*
- **Audit Finding:**
  - `Voltage` has negligible attribution ($-0.0035$ path-length units), rounding to `-0.00` in the text. Citing a zero-impact feature as a "driving feature" is an artifact of the fixed top-3 heuristic.
  - **Overall Status:** **WARNING**.

### Problematic Case 3: Speculative End-Use Hypothesis for Unmetered Residuals
- **Timestamp:** `2010-02-05 06:45:00`
- **Consensus Level:** Isolated Flag (LSTM_Prediction)
- **Diagnostic Text Excerpt:**
  > *"Sub-circuit telemetry: Sub-metering channels (Sub1: 0.0 Wh, Sub2: 0.0 Wh, Sub3: 8.0 Wh) remained relatively low or near typical levels, suggesting that the primary load deviation derived from unmetered general circuits, lighting, or wall sockets."*
- **Audit Finding:**
  - **Audit E: WARNING** (`unverified_general_circuit_hypothesis`). Sub-meter readings indicate that the surge occurred in residual circuits ($P - \sum \text{Sub}_k$). However, attributing this to "lighting, or wall sockets" without dedicated telemetry is an unverified hypothesis.
  - **Overall Status:** **WARNING**.

---

## 7. Scientific Limitations & Technical Recommendations

Based on this comprehensive audit, Member 4 documents the following limitations and recommendations for future iterations:

1. **Replace Fixed Top-K Feature Slicing with Dynamic Attribution Thresholding:**
   - *Limitation:* Taking `feat_order[:3]` forces the inclusion of stabilizing ($\phi < 0$) or negligible ($|\phi| < 0.05$) features when only 1 or 2 features drove the anomaly.
   - *Recommendation:* Filter features dynamically using a minimum positive attribution threshold:
     $$\mathcal{F}_{\text{driving}} = \{j \in \{1, \dots, M\} : \phi_j > 0.05\}$$
     If fewer than 3 features meet the threshold, only report the qualifying features rather than padding with stabilizing features.
2. **Explicitly Separate Driving Features from Normalizing Features:**
   - *Limitation:* Including stabilizing features under the header `"Key driving features: ... acted toward normalcy"` creates semantic tension.
   - *Recommendation:* Structure explanations into two distinct clauses:
     - *"Key driving features pushing anomaly score higher: Feature A (+0.85), Feature B (+0.70)."*
     - *"Stabilizing features acting toward normalcy: Feature C (-0.25)."*
3. **Restrict Residual Circuit Descriptions to Mathematical Statements:**
   - *Limitation:* Mentions of "lighting, or wall sockets" are plausible based on household circuitry, but lack sensor verification.
   - *Recommendation:* Refine the fallback template to state:
     *"Sub-metered branch circuits remained near baseline levels, indicating that the load deviation was concentrated in unmetered electrical branch circuits."*

---

## 8. Deliverables & Artifact Manifest

All deliverables for Experiment 6 are stored under [`experiments/phase3_explanation_semantic_audit/`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_explanation_semantic_audit/):

| Deliverable | File Path | Shape / Size | Purpose |
| :--- | :--- | :---: | :--- |
| **Audit Matrix CSV** | [`explanation_semantic_audit.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_explanation_semantic_audit/explanation_semantic_audit.csv) | 333 rows $\times$ 12 cols (59.6 KB) | Instance-by-instance audit record across all 5 dimensions and overall status. |
| **Quality Summary JSON** | [`explanation_quality_summary.json`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_explanation_semantic_audit/explanation_quality_summary.json) | 49 lines (1.6 KB) | Machine-readable aggregate metrics, dimension counts, and issue categories. |
| **Flagged Examples CSV** | [`explanation_issue_examples.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_explanation_semantic_audit/explanation_issue_examples.csv) | 78 rows $\times$ 6 cols (21.4 KB) | Every flagged explanation with exact excerpt, issue category, and reason. |
| **Execution Script** | [`run_explanation_semantic_audit.py`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_explanation_semantic_audit/run_explanation_semantic_audit.py) | 485 lines (21.8 KB) | Deterministic, read-only Python script executing all audit assertions. |
| **Execution Log** | [`execution.log`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_explanation_semantic_audit/execution.log) | 52 lines (4.1 KB) | Full execution trace and runtime verification log. |
| **Automated Tests** | [`explanation_audit_tests.py`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_explanation_semantic_audit/explanation_audit_tests.py) | 125 lines (5.8 KB) | 9 automated pytest validation tests. |
| **Pytest Suite** | [`test_explanation_semantic_audit.py`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/tests/test_explanation_semantic_audit.py) | 125 lines (5.8 KB) | Regression test integrated into the master test runner. |
| **Markdown Report** | [`explanation_quality_report.md`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_explanation_semantic_audit/explanation_quality_report.md) | This document | Comprehensive scientific audit report. |

---

## 9. Verification & Safeguards Compliance

1. **Master Research Paper Strict Freeze:**
   - File Path: [`research_paper/complete_ieee_paper.md`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/research_paper/complete_ieee_paper.md)
   - Expected Byte Count: `122,229 bytes`
   - Verified Byte Count: `122,229 bytes`
   - Freeze Status: **STRICTLY PRESERVED (0 bytes modified)**
2. **Production Pipeline & Output Preservation:**
   - Original [`explanations.json`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/outputs/explanations.json): 767,346 bytes (unmodified).
   - Original [`contextual_anomalies.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/Anomaly-Detection-main/outputs/contextual_anomalies.csv): 299,464 bytes (unmodified).
   - Original model weights and detection CSVs: completely untouched.
3. **Comprehensive Regression Test Suite:**
   ```bash
   python -m pytest -v tests/
   # Result: 45 passed in 7.11s (100% pass rate)
   ```

---

## 10. Conclusion

Experiment 6 (`EXP-PHASE3-XAI-06`) delivers an exhaustive, transparent scientific audit of Member 4's explanation generation pipeline:
- **No Fatal Errors:** Zero explanations failed structural integrity, consensus alignment, contextual reference rules, or scientific boundaries.
- **260 Explanations Passed All Five Dimensions Cleanly ($78.08\%$).**
- **73 Explanations Received Documented Warnings ($21.92\%$):**
  - 5 instances of negative or negligible SHAP features cited due to mechanical top-k sorting.
  - 73 instances of cautious but unmeasured end-use hypotheses regarding unmetered general circuits.
- The audit demonstrates that the XAI pipeline produces reproducible, structurally intact, and scientifically grounded explanations while honestly exposing subtle edge-case heuristics for future refinement.

