# Member 4 — Phase 3: Experimental Validation of Context-Aware XAI
## Experiment 5: Scientific Terminology, Units, and Mathematical Audit Report

**Author:** Member 4 — Explainable AI (XAI), Context-Aware Historical Twin Matching, and Research Experiments  
**Experiment Identifier:** `EXP-PHASE3-SHAP-05`  
**Audit Document:** `phase3_shap_dependence_audit_report.md`  
**Date of Audit:** 2026-11-07  
**Status:** Audit Completed, Corrected Visual Artifacts Generated, Test Suite Verified (36/36 Unit Tests Passing)  
**Strict Safeguard Compliance:**
- Master conference paper draft remains strictly **frozen** (`research_paper/complete_ieee_paper.md` verified at exactly **122,229 bytes**).
- Production anomaly detection algorithms and trained model weights unmodified.
- Historical twin matching methodology preserved.
- Original experiment data, CSV files, and original PNG figures completely preserved; corrected figures created separately.

---

## 1. Executive Summary & Audit Mandate

During Phase 3 Experiment 5 (`EXP-PHASE3-SHAP-05`), Member 4 conducted an in-depth audit of feature dependencies and TreeSHAP attribution redundancy in the fitted 200-tree Isolation Forest. While the empirical results successfully demonstrated that collinearity between bulk active power and current intensity splits tree-isolation attributions, a rigorous review of scientific precision identified several areas requiring formal terminology, unit, and conceptual correction:

1. **Dimensionality & Units:** Clarifying that SHAP values quantify inverted tree search depth ($\Delta \bar{h}$), and thus have **path-length attribution units**, not electrical power units ($\text{kW}$) or current units ($\text{A}$).
2. **Mathematical Distinction in Grouped Attributions:** Formalizing $\phi_{\text{bulk}} = \phi_{P} + \phi_{I}$ as a **post-hoc grouped attribution** and strictly differentiating the mean absolute value of summed per-instance attributions ($1.0540\text{ path-length units}$) from the arithmetic sum of individual mean absolute attributions ($1.0561\text{ path-length units}$) via the triangle inequality.
3. **Correlation vs. Independence vs. Routine Exclusivity:** Qualifying correlation statements to avoid confusing lack of linear/monotonic correlation with broad statistical independence, and removing any claim that empirical negative telemetry correlation proves mutually exclusive household routines.
4. **Grouped Importance as Descriptive Model Partitioning:** Explicitly stating that grouped importance percentages represent descriptive summary metrics of ensemble split allocation under collinearity, rather than physical causal shares of electrical consumption.

This audit report documents all required corrections, provides the mathematical derivations, reviews the updated artifact manifest, and confirms that all project safeguards remain intact.

---

## 2. Section 1: Dimensionality and Unit Audit

### 2.1 Theoretical Basis of TreeSHAP Output Space
In tree-based anomaly detection using Isolation Forests, anomaly severity is determined by how quickly an instance is isolated across an ensemble of randomized trees.
- For an instance $x$, the search depth in tree $t$ is $h_t(x)$, representing the number of edges traversed from the root node to the terminating leaf node.
- The average search depth across $T$ trees is:
  $$\bar{h}(x) = \frac{1}{T} \sum_{t=1}^T h_t(x)$$
- Shorter path lengths ($\bar{h}(x) \ll c(n)$) indicate acute anomalous isolation, whereas longer path lengths indicate typical, well-clustered instances.
- TreeSHAP explains the model's prediction function $f(x)$, which is defined in tree-depth space as the inverted average search depth (or path-length deviation from expectation):
  $$f(x) - \mathbb{E}[f(X)] = \sum_{j=1}^M \phi_j(x)$$
  where $\phi_j(x)$ is the local Shapley attribution for feature $j$.

### 2.2 Identification and Rectification of Unit Misconceptions
- **Dimensionality:** Because the model output $f(x)$ is measured in tree levels/search depth, every local attribution $\phi_j(x)$ is denominated in **dimensionless tree-depth or path-length attribution units**.
- **Physical Feature Inputs vs. Attribution Units:**
  - `Global_active_power` is recorded in kilowatts ($\text{kW}$).
  - `Global_intensity` is recorded in amperes ($\text{A}$).
  - `Voltage` is recorded in volts ($\text{V}$).
  - Sub-metering channels are recorded in watt-hours of active energy per minute ($\text{Wh/min}$).
- **The Dimensional Fallacy:** It is a common misconception in applied machine learning to assume that a feature's SHAP value inherits the physical units of that feature's input space (e.g., assuming $\phi_{\text{Global\_active\_power}}$ is measured in $\text{kW}$).
- **Audit Rectification:**
  - All SHAP attributions in Experiment 5 are explicitly designated as **"path-length attribution units"** (or **"path-length units"**).
  - No text, table header, or figure label may refer to SHAP attributions in $\text{kW}$ or $\text{A}$.
  - Axis labels in diagnostic displays have been updated to explicitly state:
    `"Mean |SHAP Attribution| (path-length units)"` and `"SHAP Anomaly Attribution: Global_active_power (path-length units)"`.

---

## 3. Section 2: Mathematical Formalization of Post-Hoc Grouped Attribution

### 3.1 Problem Definition: Collinear Attribution Splitting
In the raw telemetry, `Global_active_power` ($P$) and `Global_intensity` ($I$) have near-perfect collinearity ($r = 0.9994$ in training, $r = 0.9993$ in candidate anomalies) due to the physical Ohm's/Joule's relationship $P \approx V \cdot I$ with tightly regulated mains voltage ($\approx 240\text{V}$).

Because the Isolation Forest algorithm selects random subsets of features at each node split, the total isolation signal associated with bulk electrical draw is randomly split between $P$ and $I$. In Table IX of the baseline paper, this causes:
- `Sub_metering_2` to rank #1 individually (Mean $|SHAP| = 0.7644$ path-length units)
- `Global_intensity` to rank #3 individually ($0.5600$ path-length units)
- `Global_active_power` to rank #5 individually ($0.4961$ path-length units)

### 3.2 Formal Definition of Post-Hoc Grouped Attribution
To assess the aggregate isolation signal of bulk electrical draw without modifying the trained model or retrying tree induction, we define a **post-hoc grouped attribution** at the instance level:
$$\phi_{i, \text{bulk}} \equiv \phi_{i, P} + \phi_{i, I} \quad \text{for } i \in \{1, \dots, N\}$$

### 3.3 The Two Mathematically Distinct Metrics
We explicitly distinguish two different summary statistics evaluated across the $N = 333$ candidate anomalies:

#### Metric A: Mean Absolute Value of Summed Per-Instance Attributions
$$\overline{|\phi_{\text{bulk}}|}_{N} = \frac{1}{N}\sum_{i=1}^N |\phi_{i, \text{bulk}}| = \frac{1}{N}\sum_{i=1}^N |\phi_{i, P} + \phi_{i, I}| = \mathbf{1.0540}\text{ path-length units}$$
This metric represents the actual mean magnitude of the joint bulk power attribution after accounting for instance-level directional signs.

#### Metric B: Arithmetic Sum of Individual Mean Absolute Attributions
$$\overline{|\phi_{P}|}_{N} + \overline{|\phi_{I}|}_{N} = \left(\frac{1}{N}\sum_{i=1}^N |\phi_{i, P}|\right) + \left(\frac{1}{N}\sum_{i=1}^N |\phi_{i, I}|\right) = 0.4961 + 0.5600 = \mathbf{1.0561}\text{ path-length units}$$
This metric is the marginal sum of the individual feature rankings reported in Table IX.

### 3.4 Triangle Inequality Derivation and Empirical Divergence
By the fundamental triangle inequality for real numbers:
$$|\phi_{i, P} + \phi_{i, I}| \le |\phi_{i, P}| + |\phi_{i, I}| \quad \forall i$$
Averaging across all $N$ instances:
$$\frac{1}{N}\sum_{i=1}^N |\phi_{i, P} + \phi_{i, I}| \le \frac{1}{N}\sum_{i=1}^N |\phi_{i, P}| + \frac{1}{N}\sum_{i=1}^N |\phi_{i, I}|$$

Equality holds if and only if $\text{sign}(\phi_{i, P}) = \text{sign}(\phi_{i, I})$ for all instances where both values are non-zero.

#### Empirical Divergence Breakdown ($N = 333$ Candidate Anomalies):
- **Sign Concordance:** In **$308$ out of $333$ instances ($92.49\%$)**, both features pushed isolation depth in the identical direction:
  $$\text{sign}(\phi_{i, P}) = \text{sign}(\phi_{i, I}) \implies |\phi_{i, P} + \phi_{i, I}| = |\phi_{i, P}| + |\phi_{i, I}|$$
- **Sign Discordance (Cancellation):** In **$25$ out of $333$ instances ($7.51\%$)**, the tree splits assigned opposite attribution signs:
  $$\text{sign}(\phi_{i, P}) \ne \text{sign}(\phi_{i, I}) \implies |\phi_{i, P} + \phi_{i, I}| < |\phi_{i, P}| + |\phi_{i, I}|$$
- **Exact Numerical Discrepancy:**
  $$\Delta = (\overline{|\phi_{P}|} + \overline{|\phi_{I}|}) - \overline{|\phi_{\text{bulk}}|} = 1.0561 - 1.0540 = \mathbf{0.0021}\text{ path-length units}$$

#### Audit Ruling:
Metric A ($1.0540$) and Metric B ($1.0561$) represent mathematically distinct aggregation operations. In all future documentation and scientific reports, they must be clearly distinguished and never conflated as identical.

---

## 4. Section 3: Scientific Qualification of Correlation Limitations

### 4.1 Un-correlatedness vs. Statistical Independence
- **Issue Identified:** Prior descriptions referred to sub-metering circuits or features with low correlation as having "statistical independence confirmed" or "complete statistical orthogonality."
- **Statistical Correction:**
  - Pearson's correlation coefficient $r$ measures linear association; Spearman's rank coefficient $\rho$ measures monotonic association.
  - A near-zero correlation ($r \approx 0, \rho \approx 0$) demonstrates the **absence of linear or monotonic coupling** within the observed telemetry range.
  - It does **not** establish general statistical independence:
    $$P(X, Y) = P(X)P(Y)$$
    because nonlinear dependencies, conditional variance relationships, or higher-order joint moments may still exist.
  - Similarly, "orthogonality" is a geometric property in an inner product space, not a synonym for uncorrelated scalar time-series.
- **Audited Language:**
  - Replace "statistical independence confirmed" with:
    `"Absence of linear and monotonic correlation in baseline training telemetry (r <= 0.1390)."`
  - Replace "complete statistical orthogonality" with:
    `"Absence of linear or monotonic correlation (|r| < 0.05)."`

### 4.2 Rejection of Household Routine Exclusivity Claims
- **Issue Identified:** In candidate anomalies, `Sub_metering_1` (kitchen) and `Sub_metering_2` (laundry) exhibit an empirical negative correlation of $r = -0.3128$. Prior text asserted that this "proves mutually exclusive household routines."
- **Scientific Correction:**
  - The empirical coefficient $r = -0.3128$ simply reflects that, within the 333 candidate anomaly timestamps (which represent acute excursions), peak telemetry readings on the laundry branch rarely coincided with peak readings on the kitchen branch in the identical 15-minute sampling interval.
  - Correlation across observational telemetry does not prove causality or behavioral exclusivity. Occupants could run appliances concurrently without triggering simultaneous acute threshold excursions, or operational cycles may have different temporal durations.
  - The UCI dataset lacks ground-truth appliance-level activity logs.
- **Audited Language:**
  - Remove all assertions that negative correlation proves mutually exclusive routines.
  - Replace with:
    `"The negative correlation (r = -0.3128) during candidate anomalies indicates that peak telemetry excursions on the laundry branch rarely coincided with peak kitchen excursions in the same 15-minute window within this sample; this reflects sample-specific telemetry co-occurrence and does not prove that household routines are mutually exclusive."`

---

## 5. Section 4: Interpretation of Grouped Feature Importance

### 5.1 Descriptive Summary Metric vs. Causal Physical Shares
When evaluating the 6 functional domains:
1. Sub-metering Measurements (3 features): $1.4288$ path-length units ($27.14\%$)
2. Direct Electrical Measurements (4 features): $1.3627$ path-length units ($25.89\%$)
3. Rolling-Window Features (2 features): $0.9719$ path-length units ($18.46\%$)
4. Lag Features (2 features): $0.6039$ path-length units ($11.47\%$)
5. Temporal / Calendar Features (4 features): $0.4713$ path-length units ($8.95\%$)
6. Consumption-Change Features (1 feature): $0.4252$ path-length units ($8.08\%$)

- **Scientific Clarification:**
  - These percentage shares represent **descriptive summary metrics** describing how the trained 200-tree Isolation Forest distributed its split criteria across feature groups when isolating candidate anomalies.
  - Because tree induction randomly samples feature subspaces at each split node, the presence of collinear and correlated features redistributes split opportunities across the ensemble.
  - These percentages **do not represent physical causal shares** of household electrical load, total energy consumption, or power distribution.
  - In all documentation, these values are designated as "descriptive percentage of total anomaly attribution" or "model partitioning share."

---

## 6. Section 5: Figure Audit & Corrected Visual Artifacts

To adhere strictly to the rule:
*"Preserve original CSVs and figures; create corrected versions separately if required."*

Member 4 created a dedicated script [`generate_corrected_figures.py`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/generate_corrected_figures.py) which preserves the original figures and outputs corrected publication figures:

| Figure Name | Original Artifact | Corrected Artifact | Audit Corrections Applied |
| :--- | :--- | :--- | :--- |
| **Figure 1: Feature Dependence & Attribution Redundancy** | [`feature_dependence_analysis.png`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/feature_dependence_analysis.png) | [`feature_dependence_analysis_corrected.png`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/feature_dependence_analysis_corrected.png) | 1. Panel D axis labels explicitly designated with `(path-length units)`.<br>2. Explanatory footnote added: *"Values quantify tree-depth search reduction (path-length units), not kW or A."*<br>3. Panel titles and labels audited for precision. |
| **Figure 2: Grouped Importance & Post-Hoc Consolidation** | [`grouped_shap_importance.png`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/grouped_shap_importance.png) | [`grouped_shap_importance_corrected.png`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_shap_dependence/grouped_shap_importance_corrected.png) | 1. Panel A title updated to *"Descriptive Grouped Importance (Ensemble Split Share)"* with footnote on descriptive model partitioning vs causal shares.<br>2. Panel B y-axis explicitly labeled *"Mean \|SHAP Attribution\| (path-length units)".*<br>3. Panel B bulk bar labeled *"Post-Hoc Grouped Bulk (phi_bulk)".*<br>4. Dotted reference line added for arithmetic sum ($1.0561$).<br>5. Explicit annotation of triangle inequality discrepancy ($\Delta = 0.0021$, $7.5\%$ discordant signs). |

---

## 7. Section 6: Comprehensive Artifact and File Verification

### 7.1 Master Research Paper Safeguard Check
The master conference paper draft was inspected to ensure absolute preservation of the freeze:
- **File Path:** [`research_paper/complete_ieee_paper.md`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/research_paper/complete_ieee_paper.md)
- **Expected Byte Count:** `122,229 bytes`
- **Verified Byte Count:** `122,229 bytes`
- **Result:** **STRICT FREEZE FULLY PRESERVED (0 bytes modified)**

### 7.2 Automated Test Suite Execution
The automated test suite across all Phase 2 and Phase 3 contributions was executed:
- `tests/test_xai_pipeline.py` (8 passed)
- `tests/test_context_sensitivity.py` (7 passed)
- `tests/test_shap_stability.py` (7 passed)
- `tests/test_shap_randomized_stability.py` (7 passed)
- `tests/test_shap_dependence.py` (7 passed)
- **Overall Result:** **36 passed in 5.99 seconds (100% pass rate)**

### 7.3 Complete Experiment 5 Artifact Inventory

```
experiments/phase3_shap_dependence/
├── execution.log                                     [Runtime execution log]
├── feature_correlation_pairs.csv                     [120 unique pairs with Raw & SHAP correlations]
├── feature_value_correlations.csv                    [256-pair raw feature correlation matrix]
├── shap_attribution_correlations.csv                 [256-pair SHAP attribution correlation matrix]
├── grouped_shap_importance.csv                       [6 functional domain grouped metrics]
├── feature_dependence_analysis.png                   [Original Figure 1 - Preserved]
├── feature_dependence_analysis_corrected.png         [Corrected Figure 1 with path-length units]
├── grouped_shap_importance.png                       [Original Figure 2 - Preserved]
├── grouped_shap_importance_corrected.png             [Corrected Figure 2 with audited terminology]
├── run_phase3_shap_dependence_experiment.py          [Original execution script]
├── generate_corrected_figures.py                     [Corrected figure generation script]
├── phase3_shap_dependence_report.md.bak              [Original report backup - Preserved]
├── phase3_shap_dependence_report.md                  [Updated report with audited terminology]
└── phase3_shap_dependence_audit_report.md            [This comprehensive audit deliverable]
```

---

## 8. Summary of Audit Conclusions

1. **Dimensional Consistency Established:** All SHAP values operate in inverted tree-depth space $\Delta \bar{h}$ and are properly denominated in path-length attribution units. All references to physical $\text{kW}$ or $\text{A}$ units for SHAP values have been eradicated.
2. **Mathematical Distinction Enforced:** The post-hoc grouped bulk attribution $\phi_{\text{bulk}} = \phi_P + \phi_I$ is now formally defined. The distinction between the instance-level mean absolute grouped attribution ($1.0540$ path-length units) and the marginal arithmetic sum of individual means ($1.0561$ path-length units) is proven and documented via the triangle inequality across $7.5\%$ discordant instances.
3. **Statistical Restraint Applied:** Unqualified claims of "statistical independence" and "complete orthogonality" have been replaced with precise statements of un-correlatedness ($r \le 0.1390$). Claims attributing negative correlation to mutually exclusive household routines have been removed.
4. **Interpretive Humility Maintained:** Grouped feature importance is explicitly identified as a descriptive metric of tree split allocation under collinearity, not physical causal shares.
5. **Full Non-Destructive Integrity:** The research paper remains strictly frozen at 122,229 bytes, all production code and baseline results remain untouched, original CSVs and PNGs remain intact, and all 36 test invariants pass without error.

