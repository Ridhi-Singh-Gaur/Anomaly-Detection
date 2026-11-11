# Member 4 — Phase 3: Experimental Validation of Context-Aware XAI
## Experiment 5: Feature Dependence and SHAP Attribution Redundancy Audit Report

**Author:** Member 4 — Explainable AI (XAI), Context-Aware Analysis, and Research Experiments  
**Experiment Identifier:** `EXP-PHASE3-SHAP-05`  
**Execution Date:** 2026-11-07  
**Status:** Executed, Fully Verified, and Reproducible (36/36 Automated Unit Tests Passing)  
**Integrity Safeguard:** Research paper draft remains **frozen** (`complete_ieee_paper.md` verified at exactly 122,229 bytes). No production models, baseline weights, training data, or previous experimental outputs modified.

---

## 1. Executive Summary & Audit Objectives

In Phase 3 Experiments 3 and 4 (`EXP-PHASE3-SHAP-03` and `EXP-PHASE3-SHAP-04`), Member 4 established that the TreeSHAP explanations for the fitted 200-tree Isolation Forest are asymptotically stable and statistically invariant to randomized tree resampling. However, tree-based ensemble explainer algorithms—including TreeSHAP—are known to exhibit specific behaviors when input features are mutually correlated or statistically collinear.

This audit conducts a rigorous, non-destructive **Feature Dependence and SHAP Attribution Redundancy Analysis (`EXP-PHASE3-SHAP-05`)** to investigate:
1. Whether the 16 features used by the Isolation Forest exhibit strong statistical dependencies in the raw telemetry data.
2. Whether correlated raw features share, redistribute, or obscure their game-theoretic TreeSHAP attributions.
3. How feature grouping across functional domains alters the apparent distribution of anomaly importance.
4. The fundamental interpretability limitations of the existing SHAP results (e.g., Table IX of the research paper).

### Scientific Unit & Terminology Definitions
- **Attribution Unit:** All SHAP values in this experiment quantify local deviations in the model's output function—specifically, the **inverted average Isolation Forest search depth** $\Delta \bar{h}(x)$. Consequently, SHAP attributions are denominated in **dimensionless tree-depth or path-length attribution units**, not physical electrical units such as kilowatts ($\text{kW}$) or amperes ($\text{A}$).
- **Post-Hoc Grouped Attribution:** The joint contribution of `Global_active_power` and `Global_intensity` is defined post-hoc at the instance level as $\phi_{i, \text{bulk}} = \phi_{i, P} + \phi_{i, I}$.
- **Descriptive Grouped Importance:** Group-level percentage sums are descriptive summary metrics that reflect empirical tree split allocation across features under collinearity, not physical causal shares of power consumption.

### Key Empirical Findings
1. **Collinear Redundancy in Bulk Electrical Features:**
   - `Global_active_power` and `Global_intensity` exhibit near-perfect statistical collinearity in raw observations:
     - Training partition ($N = 109,347$): Pearson $r = 0.9994$, Spearman $\rho = 0.9969$.
     - Candidate anomalies ($N = 333$): Pearson $r = 0.9993$, Spearman $\rho = 0.9987$.
   - Their TreeSHAP attributions are strongly correlated across candidate anomalies ($r = 0.9749$, $\rho = 0.9480$), with identical attribution signs in **$92.5\%$ of candidate instances** (308/333).
2. **Attribution Splitting and Post-Hoc Grouped Attribution:**
   - In individual feature rankings (Table IX), `Sub_metering_2` appears as the #1 feature (Mean $|SHAP| = 0.7644$ path-length units), while `Global_intensity` is #3 ($0.5600$) and `Global_active_power` is #5 ($0.4961$).
   - Because tree induction randomly samples feature subspaces at each split, the ensemble splits the isolation signal for bulk load between active power and current intensity.
   - When evaluated as a **post-hoc grouped attribution** ($\phi_{i, \text{bulk}} = \phi_{i, P} + \phi_{i, I}$), the mean absolute value of the summed per-instance attributions reaches:
     $$\frac{1}{N}\sum_{i=1}^N |\phi_{i, P} + \phi_{i, I}| = \mathbf{1.0540}\text{ path-length units}$$
     which exceeds `Sub_metering_2` ($0.7644$) by **$+37.9\%$**.
   - By comparison, the **arithmetic sum of individual mean absolute attributions** is:
     $$\frac{1}{N}\sum_{i=1}^N |\phi_{i, P}| + \frac{1}{N}\sum_{i=1}^N |\phi_{i, I}| = 0.4961 + 0.5600 = \mathbf{1.0561}\text{ path-length units}$$
     These two values differ by $\Delta = 0.0021$ because of the triangle inequality across the $7.5\%$ of candidate instances where tree splits assigned opposite attribution signs ($\text{sign}(\phi_P) \ne \text{sign}(\phi_I)$). They are mathematically distinct operations.
3. **Lack of Linear and Monotonic Correlation Across Sub-metering Circuits:**
   - Unlike bulk electrical measurements, the three sub-metering circuits exhibit low mutual linear and monotonic correlation in training data ($r \le 0.1390$) and modest negative correlation during candidate anomalies ($r = -0.3128$ between kitchen and laundry).
   - Their SHAP attributions exhibit minimal linear correlation ($r = -0.0367$ between laundry and HVAC; $r = 0.0166$ between kitchen and HVAC).
   - This absence of linear/monotonic correlation indicates that sub-metering channels provide distinct circuit-level telemetry streams. It does not imply broad statistical independence across higher-order interactions, nor does negative correlation prove that household routines are mutually exclusive.
4. **Descriptive Grouped Importance Across Functional Domains:**
   - Grouping the 16 features into 6 functional domains reveals the following descriptive summary shares:
     - **Sub-metering Measurements (3 features):** $1.4288$ path-length units ($27.14\%$ of total attribution sum)
     - **Direct Electrical Measurements (4 features):** $1.3627$ path-length units ($25.89\%$ of total attribution sum)
     - **Rolling-Window Features (2 features):** $0.9719$ path-length units ($18.46\%$ of total attribution sum)
     - **Lag Features (2 features):** $0.6039$ path-length units ($11.47\%$ of total attribution sum)
     - **Temporal / Calendar Features (4 features):** $0.4713$ path-length units ($8.95\%$ of total attribution sum)
     - **Consumption-Change Features (1 feature):** $0.4252$ path-length units ($8.08\%$ of total attribution sum)
   - These proportions describe how the ensemble allocates split decisions across available features in the presence of overlapping telemetry; they do not represent physical causal shares of grid load.

---

## 2. Task 1: Feature-Value Correlation Analysis

Pearson and Spearman correlation matrices were computed across four distinct partitions:
- **Training observations** ($N_{\text{train}} = 109,347$): long-term operational population baseline.
- **Candidate anomalies** ($N = 333$): the frozen multi-model consensus anomaly pool.
- **Normal observations** ($N = 100$): test observations outside the candidate pool (seed 42).
- **Combined evaluation sample** ($N = 433$): unified test evaluation set.

### 2.1 Highly Correlated Feature Pairs in the Training Dataset

| Feature 1 | Feature 2 | Train Pearson ($r$) | Train Spearman ($\rho$) | Candidate Pearson ($r$) | Candidate Spearman ($\rho$) | Correlation Tier | Physical / Structural Basis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Global_active_power** | **Global_intensity** | **+0.9994** | **+0.9969** | **+0.9993** | **+0.9987** | $|r| \ge 0.90$ | Physical coupling: $P \approx V \cdot I$ with tightly regulated mains voltage ($\approx 240\text{V}$). |
| **rolling_mean_4** | **lag_1** | **+0.9386** | **+0.9355** | **+0.8237** | **+0.8553** | $|r| \ge 0.90$ | Autoregressive structural overlap: `rolling_mean_4` embeds `lag_1` as its most recent component. |
| **Global_active_power** | **rolling_mean_4** | **+0.8806** | **+0.9032** | $+0.7937$ | $+0.8402$ | $|r| \ge 0.80$ | Autoregressive persistence across the 1-hour rolling window. |
| **Global_intensity** | **rolling_mean_4** | **+0.8777** | **+0.8978** | $+0.7925$ | $+0.8385$ | $|r| \ge 0.80$ | Intensity persistence across the 1-hour rolling window. |
| **Global_active_power** | **lag_1** | **+0.8477** | **+0.8600** | $+0.6558$ | $+0.7258$ | $|r| \ge 0.80$ | 15-minute lag-1 power persistence. |
| **Global_intensity** | **lag_1** | **+0.8450** | **+0.8535** | $+0.6552$ | $+0.7239$ | $|r| \ge 0.80$ | 15-minute lag-1 intensity persistence. |
| **day_of_week** | **is_weekend** | **+0.7898** | **+0.7891** | $+0.8171$ | $+0.8170$ | $|r| \ge 0.70$ | Calendar construction: weekend corresponds to $\text{day\_of\_week} \in \{5, 6\}$. |
| **lag_4** | **rolling_mean_4** | **+0.7633** | **+0.7987** | $+0.4746$ | $+0.5367$ | $|r| \ge 0.70$ | Trailing boundary of the 1-hour rolling average window. |
| **lag_1** | **lag_4** | $+0.6365$ | **+0.7093** | $+0.1698$ | $+0.2970$ | $|\rho| \ge 0.70$ | Autoregressive correlation decaying over a 45-minute lag interval. |

### 2.2 Target Hypothesis Pair Audit

We specifically audited the five target feature pairs identified in the experimental design:
1. **Global_active_power vs. Global_intensity:**
   - Train: $r = 0.9994$, $\rho = 0.9969$. Candidates: $r = 0.9993$, $\rho = 0.9987$.
   - **Finding:** Near-perfect physical collinearity. Both features provide redundant representations of bulk household electrical draw.
2. **rolling_mean_4 vs. lag_1:**
   - Train: $r = 0.9386$, $\rho = 0.9355$. Candidates: $r = 0.8237$, $\rho = 0.8553$.
   - **Finding:** Strong autoregressive dependency arising from structural mathematical overlap.
3. **Sub_metering_1, Sub_metering_2, and Sub_metering_3:**
   - Sub1 vs Sub2: Train $r = 0.0799$; Candidates $r = -0.3128$.
   - Sub1 vs Sub3: Train $r = 0.1390$; Candidates $r = 0.1361$.
   - Sub2 vs Sub3: Train $r = 0.0996$; Candidates $r = 0.0854$.
   - **Finding:** Absence of linear and monotonic correlation in baseline training data ($r \le 0.1390$). Sub-metering circuits operate on distinct electrical branch feeds. The negative correlation ($r = -0.3128$) during candidate anomalies reflects an empirical sample property where peak excursions on the laundry branch rarely coincided with peak excursions on the kitchen branch in the same 15-minute window; this does not prove that occupant routines are mutually exclusive.
4. **lag_1 vs. lag_4:**
   - Train: $r = 0.6365$, $\rho = 0.7093$. Candidates: $r = 0.1698$, $\rho = 0.2970$.
   - **Finding:** Moderate temporal autocorrelation under regular conditions that diminishes during acute anomalous events.
5. **rolling_std_4 vs. consumption_change:**
   - Train: $r = 0.0315$, $\rho = -0.0605$. Candidates: $r = 0.0476$, $\rho = 0.0508$.
   - **Finding:** Absence of linear or monotonic correlation ($|r| < 0.05$). While both capture dynamic behavior, `consumption_change` is a signed 1-step gradient ($x_t - x_{t-1}$), whereas `rolling_std_4` is an unsigned dispersion metric.

---

## 3. Task 2: SHAP Attribution Dependence Analysis

Using the reference 200-tree SHAP attribution matrix ($N = 433$), we evaluated Pearson and Spearman correlation matrices between feature attribution columns (denominated in path-length attribution units).

### 3.1 Comparison of Raw-Feature Correlations and SHAP Correlations

| Feature Pair | Raw Pearson (Candidates) | SHAP Pearson (Candidates) | Discrepancy ($|r_{\text{raw}} - r_{\text{shap}}|$) | SHAP Dependency Characterization |
| :--- | :---: | :---: | :---: | :--- |
| **Global_active_power vs Global_intensity** | **+0.9993** | **+0.9749** | **0.0244** | **High Attribution Collinearity:** Attributions covary in direction and magnitude ($92.5\%$ sign agreement). |
| **rolling_mean_4 vs lag_1** | **+0.8237** | **+0.6872** | $0.1365$ | **Moderate Attribution Coupling:** Historical persistence features reinforce each other in tree paths. |
| **Global_intensity vs Sub_metering_1** | $+0.4079$ | **+0.5445** | $0.1366$ | **Positive Co-Attribution:** Kitchen surges register concurrently on main current telemetry. |
| **Global_active_power vs Sub_metering_1** | $+0.4079$ | **+0.4998** | $0.0919$ | **Positive Co-Attribution:** Kitchen surges register concurrently on bulk power telemetry. |
| **Sub_metering_1 vs Sub_metering_2** | $-0.3128$ | **-0.4139** | $0.1011$ | **Negative Attribution Correlation:** Inverse excursion pattern observed in candidate anomaly sample. |
| **rolling_std_4 vs consumption_change** | $+0.0476$ | **+0.0979** | $0.0503$ | **Uncorrelated Attributions:** Volatility and 1-step gradient contribute independently in tree depth. |
| **Sub_metering_2 vs Sub_metering_3** | $+0.0854$ | **-0.0367** | $0.1221$ | **Negligible Attribution Correlation:** Laundry and HVAC circuits exhibit uncoupled attributions. |

---

## 4. Task 3: Grouped Feature Importance

To evaluate the broader distribution of anomaly attributions across functional domains, the 16 features were organized into 6 functional groups:
1. **Direct Electrical Measurements (4 features):** `Global_active_power`, `Global_intensity`, `Global_reactive_power`, `Voltage`
2. **Sub-metering Measurements (3 features):** `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`
3. **Rolling-Window Features (2 features):** `rolling_mean_4`, `rolling_std_4`
4. **Lag Features (2 features):** `lag_1`, `lag_4`
5. **Temporal / Calendar Features (4 features):** `hour`, `day_of_week`, `is_weekend`, `month`
6. **Consumption-Change Features (1 feature):** `consumption_change`

### 4.1 Group Importance Under Multiple Aggregation Norms (Candidate Anomalies, $N = 333$)

| Functional Group Name | Feature Count | $L_1$ Importance Sum (path-length units) | Descriptive % of Total Importance | Sample-wise $L_2$ Euclidean Norm | Sample-wise Net Signed Magnitude | Within-Group Feature Allocation (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sub-metering Measurements** | 3 | **$1.4288$** | **$27.14\%$** | $1.1912$ | $1.3145$ | Sub2 ($53.5\%$), Sub1 ($36.7\%$), Sub3 ($9.8\%$) |
| **Direct Electrical Measurements** | 4 | **$1.3627$** | **$25.89\%$** | $0.8333$ | $1.2411$ | Intensity ($41.1\%$), Active ($36.4\%$), Reactive ($14.3\%$), Voltage ($8.2\%$) |
| **Rolling-Window Features** | 2 | **$0.9719$** | **$18.46\%$** | $0.7841$ | $0.9681$ | rolling_std_4 ($67.7\%$), rolling_mean_4 ($32.3\%$) |
| **Lag Features** | 2 | **$0.6039$** | **$11.47\%$** | $0.5233$ | $0.5769$ | lag_1 ($56.3\%$), lag_4 ($43.7\%$) |
| **Temporal / Calendar Features** | 4 | **$0.4713$** | **$8.95\%$** | $0.2906$ | $0.2592$ | weekend ($47.1\%$), day_of_week ($23.1\%$), hour ($15.5\%$), month ($14.3\%$) |
| **Consumption-Change Features** | 1 | **$0.4252$** | **$8.08\%$** | $0.4252$ | $0.4252$ | consumption_change ($100.0\%$) |
| **Total Benchmark Importance** | **16** | **$5.2637$** | **$100.00\%$** | — | — | — |

*Methodological Clarification:*
- The $L_1$ importance sum represents the descriptive arithmetic sum of individual mean absolute SHAP values within each domain.
- The grouped absolute SHAP importance percentages are **descriptive summary metrics** that describe how the Isolation Forest ensemble distributes its tree split criteria across feature groups in the presence of overlapping telemetry. They do not quantify physical causal shares of electrical consumption.

---

## 5. Task 4: Robustness Check & Post-Hoc Grouped Attribution

### 5.1 The Collinear Attribution Splitting Effect
When two features $X_1$ and $X_2$ are highly collinear ($r = 0.9994$), tree-building algorithms randomly select between them at split nodes based on minor sampling variations. In TreeSHAP, this causes the total isolation signal associated with that physical dimension to be split across both features.

To evaluate this effect without altering the trained model, we define a **post-hoc grouped attribution** for bulk electrical draw:
$$\phi_{i, \text{bulk}} \equiv \phi_{i, \text{Global\_active\_power}} + \phi_{i, \text{Global\_intensity}}$$

### 5.2 Mathematical Distinction Between Grouped Attribution and Arithmetic Sum of Means

We must strictly distinguish two distinct mathematical quantities:
1. **Mean absolute value of summed per-instance attributions:**
   $$\overline{|\phi_{\text{bulk}}|}_{N} = \frac{1}{N}\sum_{i=1}^N |\phi_{i, P} + \phi_{i, I}| = \mathbf{1.0540}\text{ path-length units}$$
2. **Arithmetic sum of individual mean absolute attributions:**
   $$\overline{|\phi_{P}|}_{N} + \overline{|\phi_{I}|}_{N} = \frac{1}{N}\sum_{i=1}^N |\phi_{i, P}| + \frac{1}{N}\sum_{i=1}^N |\phi_{i, I}| = 0.4961 + 0.5600 = \mathbf{1.0561}\text{ path-length units}$$

By the triangle inequality:
$$\frac{1}{N}\sum_{i=1}^N |\phi_{i, P} + \phi_{i, I}| \le \frac{1}{N}\sum_{i=1}^N |\phi_{i, P}| + \frac{1}{N}\sum_{i=1}^N |\phi_{i, I}|$$

| Metric | Individual Global_intensity | Individual Global_active_power | Individual Sub_metering_2 | Post-Hoc Grouped Bulk ($\phi_{\text{bulk}}$) | Arithmetic Sum ($\overline{|\phi_P|} + \overline{|\phi_I|}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Absolute Attribution** | $0.5600$ path-length units | $0.4961$ path-length units | $0.7644$ path-length units | **$1.0540$ path-length units** | **$1.0561$ path-length units** |
| **Table IX Baseline Rank** | Rank 3 | Rank 5 | **Rank 1** | **Rank 1 (Grouped)** | — |
| **Sign Agreement Rate** | — | — | — | **$92.5\%$** (308/333 candidates) | — |
| **Discrepancy ($\Delta$)** | — | — | — | **$\Delta = 0.0021$** (triangle inequality) | — |

*Key Findings:*
- In $92.5\%$ of candidate instances (308/333), the signs of $\phi_P$ and $\phi_I$ coincide ($\text{sign}(\phi_P) = \text{sign}(\phi_I)$).
- In the remaining $7.5\%$ of instances (25/333), the attribution signs oppose each other, causing local cancellation in $|\phi_P + \phi_I| < |\phi_P| + |\phi_I|$. This creates the exact difference $\Delta = 0.0021$ path-length units.
- Under post-hoc grouping ($\phi_{\text{bulk}} = 1.0540$ path-length units), bulk electrical draw exceeds `Sub_metering_2` ($0.7644$) by **$+37.9\%$**.
- `Sub_metering_2` ranks #1 in Table IX individually because its telemetry is concentrated in a single feature without a collinear twin dividing its tree splits.

---

## 6. Visualizations & Diagnostic Displays

All figures are stored under [`experiments/phase3_shap_dependence/`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence). The original generated figures are preserved, and corrected figures incorporating scientific unit labels and audited terminology are provided:

### 6.1 Figure 1: Feature Dependence & SHAP Attribution Redundancy
- **Original Figure:** [`feature_dependence_analysis.png`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/feature_dependence_analysis.png)
- **Corrected Figure:** [`feature_dependence_analysis_corrected.png`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/feature_dependence_analysis_corrected.png)

*Figure Details (Corrected):*
- **Panel A (Raw Feature Correlation Heatmap):** Demonstrates pairwise Pearson correlation across all 16 features for candidate anomalies ($N=333$), highlighting collinear bulk power and intensity ($r = 0.9993$).
- **Panel B (TreeSHAP Attribution Correlation Heatmap):** Demonstrates that attribution correlation mirrors raw correlation for collinear features ($r = 0.9749$), while sub-metering circuits remain largely uncorrelated.
- **Panel C (Raw vs. SHAP Correlation Discrepancy):** Evaluates all 120 unique feature pairs against the identity line ($r_{\text{SHAP}} = r_{\text{Raw}}$).
- **Panel D (Attribution Collinearity Scatter):** Displays instance-level SHAP attributions for `Global_active_power` vs. `Global_intensity` explicitly denominated in **path-length units**, with explanatory caption confirming that values represent inverted tree search depth, not $\text{kW}$ or $\text{A}$.

### 6.2 Figure 2: Grouped Feature Importance & Post-Hoc Consolidation Analysis
- **Original Figure:** [`grouped_shap_importance.png`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/grouped_shap_importance.png)
- **Corrected Figure:** [`grouped_shap_importance_corrected.png`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/grouped_shap_importance_corrected.png)

*Figure Details (Corrected):*
- **Panel A (Descriptive Grouped Importance):** Displays the descriptive percentage share of anomaly attribution across the 6 functional domains, with a footnote clarifying that these percentages reflect empirical ensemble split allocation rather than physical causal energy shares.
- **Panel B (Post-Hoc Grouped Attribution vs. Individual Split Attributions):** Contrasts individual feature ranks against the post-hoc grouped bulk draw ($\phi_{\text{bulk}} = 1.0540$ path-length units), with explicit reference lines for `Sub_metering_2` ($0.7644$) and the arithmetic sum of individual means ($1.0561$ path-length units), accompanied by annotation of the triangle inequality discrepancy ($\Delta = 0.0021$).

---

## 7. Scientific Guardrails & Interpretability Limitations

1. **Dimensionality of TreeSHAP Attributions:**
   - TreeSHAP attributions explain the model's output function, which in Isolation Forest is the inverted average search depth $\Delta \bar{h}(x)$.
   - All SHAP attributions are denominated in **dimensionless tree-depth or path-length attribution units**.
   - Under no circumstances should SHAP attributions for `Global_active_power` be reported in $\text{kW}$, or `Global_intensity` in $\text{A}$.
2. **Correlation Between SHAP Values Is Not Physical Causality:**
   - The high attribution correlation between `shap_Global_active_power` and `shap_Global_intensity` ($r = 0.9749$) is a consequence of joint tree partitioning on collinear inputs, not a causal relationship between variables.
3. **Correlation Limitations and Routine Exclusivity:**
   - Near-zero Pearson or Spearman correlation indicates the absence of linear or monotonic co-variation in the recorded telemetry. It does not establish complete statistical independence across higher-order moments.
   - The empirical negative correlation between kitchen and laundry telemetry during candidate anomalies ($r = -0.3128$) indicates that excursions on these two circuits rarely coincided in the same 15-minute window in this dataset; it does **not** prove that household routines or appliance operations are physically mutually exclusive.
4. **Grouped Importance as Descriptive Model Metrics:**
   - Grouped attribution shares (e.g., Sub-metering $27.14\%$, Direct Electrical $25.89\%$) describe how the ensemble partitions its decision space under feature redundancy. They are descriptive summary metrics, not physical causal allocations of grid power.
5. **Circuit Telemetry vs. Appliance Identification:**
   - Sub-metering channels represent branch electrical circuits, not individual appliances. Explanations must not claim specific appliance operations (e.g., dishwashers, microwaves, or laundry cycles).
6. **Preservation of Master Manuscript:**
   - The master research paper ([`complete_ieee_paper.md`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/research_paper/complete_ieee_paper.md)) remains **strictly frozen at exactly 122,229 bytes**.

---

## 8. Deliverables & Artifact Manifest

All artifacts for Experiment 5 are located under [`experiments/phase3_shap_dependence/`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence):

| Deliverable | File Path | Shape / Size | Purpose |
| :--- | :--- | :---: | :--- |
| **Raw Correlation CSV** | [`feature_value_correlations.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/feature_value_correlations.csv) | 256 rows $\times$ 10 cols (26.6 KB) | $16 \times 16$ raw feature correlation matrices across 4 partitions. |
| **SHAP Correlation CSV** | [`shap_attribution_correlations.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/shap_attribution_correlations.csv) | 256 rows $\times$ 8 cols (23.9 KB) | $16 \times 16$ TreeSHAP attribution correlations across partitions. |
| **Pair Comparison CSV** | [`feature_correlation_pairs.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/feature_correlation_pairs.csv) | 120 rows $\times$ 19 cols (18.8 KB) | 120 unique pairs with Raw vs SHAP correlations and discrepancies. |
| **Grouped Importance CSV** | [`grouped_shap_importance.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/grouped_shap_importance.csv) | 6 rows $\times$ 8 cols (1.8 KB) | 6 functional groups with $L_1$, $L_2$, net signed sums, and within-group shares. |
| **Figure 1 (Original)** | [`feature_dependence_analysis.png`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/feature_dependence_analysis.png) | 480.5 KB | Original 4-panel diagnostic figure. |
| **Figure 1 (Corrected)** | [`feature_dependence_analysis_corrected.png`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/feature_dependence_analysis_corrected.png) | 522 KB | Corrected 4-panel figure with explicit path-length unit labels. |
| **Figure 2 (Original)** | [`grouped_shap_importance.png`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/grouped_shap_importance.png) | 215.3 KB | Original bar charts of functional groups and bulk consolidation. |
| **Figure 2 (Corrected)** | [`grouped_shap_importance_corrected.png`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/grouped_shap_importance_corrected.png) | 237 KB | Corrected bar charts with audited terminology and unit labels. |
| **Figure Script** | [`generate_corrected_figures.py`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/generate_corrected_figures.py) | 145 lines | Deterministic script generating corrected publication figures. |
| **Execution Script** | [`run_phase3_shap_dependence_experiment.py`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/run_phase3_shap_dependence_experiment.py) | 489 lines | Deterministic experiment execution script. |
| **Execution Log** | [`execution.log`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/execution.log) | 55 lines (4.2 KB) | Complete runtime verification trace. |
| **Automated Tests** | [`test_shap_dependence.py`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/tests/test_shap_dependence.py) | 111 lines | Pytest test suite asserting 7 structural invariants. |
| **Audited Report** | [`phase3_shap_dependence_report.md`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/phase3_shap_dependence_report.md) | This document | Fully audited report with corrected units and terminology. |
| **Original Report Backup** | [`phase3_shap_dependence_report.md.bak`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/phase3_shap_dependence_report.md.bak) | 225 lines | Preserved original version of the report. |

---

## 9. Conclusion

Experiment 5 (`EXP-PHASE3-SHAP-05`) establishes four key scientific conclusions:
1. **TreeSHAP Attributions Operate in Path-Length Units:** All attributions explain inverted average tree search depth ($\Delta \bar{h}$), and are properly denominated in path-length attribution units, not physical power in $\text{kW}$.
2. **Post-Hoc Grouped Attribution Reveals Bulk Power Dominance:** Bulk power and intensity split the ensemble's isolation signal due to collinearity. Under post-hoc grouping ($\phi_{\text{bulk}} = \phi_P + \phi_I$), mean absolute attribution reaches $1.0540$ path-length units ($+37.9\%$ over `Sub_metering_2`), distinct from the arithmetic sum of individual means ($1.0561$) due to the triangle inequality.
3. **Absence of Linear Correlation Is Not Complete Independence:** Sub-metering circuits exhibit low linear and monotonic correlation ($r \le 0.1390$), indicating uncoupled telemetry; negative correlation during anomalies does not imply mutually exclusive occupant routines.
4. **Grouped Importance Represents Descriptive Model Partitioning:** Group importance shares describe tree split allocation under collinearity, not physical causal shares of grid load.
5. **Research Paper Freeze Preserved:** The master manuscript remains untouched at exactly 122,229 bytes.
