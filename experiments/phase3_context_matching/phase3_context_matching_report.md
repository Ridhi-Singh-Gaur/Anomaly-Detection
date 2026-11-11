# Member 4 — Phase 3: Experimental Validation of Context-Aware XAI
## Comparative Evaluation of Three Contextual Reference Strategies

**Author:** Member 4 — Explainable AI (XAI), Context-Aware Analysis, and Research Experiments  
**Experiment Identifier:** `EXP-PHASE3-CONTEXT-01`  
**Date:** 2026-11-07  
**Status:** Executed, Fully Verified, and Reproducible  
**Integrity Safeguard:** Research paper draft remains frozen. Zero paper or baseline files modified.

---

## 1. Executive Summary & Experimental Purpose

The core premise of Member 4's research contribution is that anomaly detection alerts from complex black-box models (such as Isolation Forests and deep LSTM networks) lack operational semantics unless grounded against an appropriate historical baseline. However, the choice of contextual reference strategy fundamentally dictates the diagnostic interpretation of whether a load surge is an expected household routine or a genuine operational outlier.

This experiment conducts a rigorous, empirical comparison across **three contextual reference strategies** on the exact, frozen candidate anomaly pool ($N = 333$ candidate timestamps) derived from the 9.7-month test period ($N_{\text{test}} = 27,337$):
1. **Strategy A (Global Historical Median):** Uses all eligible historical training observations ($N_{\text{train}} = 109,347$) to establish a static, context-blind population baseline.
2. **Strategy B (Hour-Based and Weekend-Based Baseline):** Uses a 48-bin lookup table ($24\text{ hours} \times 2\text{ weekend states}$) established by the classical statistical baseline (Members 1–3).
3. **Strategy C (Dynamic Historical Twin Matching):** Uses Member 4's multi-dimensional twin cohort retrieval ($h_t \pm 1\text{ hr} \pmod{24}$, $w_k = w_t$, $m_t \pm 1\text{ mo} \pmod{12}$, fallback rule if $N < 50$), strictly referencing training data without lookahead leakage.

### Key Empirical Findings
- **Deviation Shrinkage & Contextual Grounding:** Dynamic Twin Matching (Strategy C) reduces the mean absolute contextual deviation from **$2.7412\text{ kW}$** (Global) to **$2.2097\text{ kW}$** (Dynamic), reflecting a **$19.4\%$ reduction** in baseline distortion.
- **Seasonal Climate Accommodation:** In peak winter (February), heating and earlier domestic routines elevate typical evening consumption to $2.46\text{ kW}$, whereas the static year-round hour-weekend baseline (Strategy B) assumes a median of $1.58\text{ kW}$ (a $+0.88\text{ kW}$ divergence). Conversely, in late summer (September), Strategy B overestimates typical evening load by $+0.70\text{ kW}$ ($1.51\text{ kW}$ vs. seasonal $0.81\text{ kW}$).
- **Cohort Sufficiency:** All 333 candidate anomalies across all three strategies possessed sufficient sample support ($N \ge 888$, median $N = 2,340$ for Strategy C), with zero instances triggering the $< 50$ emergency fallback.
- **Zero Lookahead Leakage:** In all three strategies, maximum reference cohort timestamps were strictly bounded by the end of the training partition ($2010\text{-}02\text{-}05\ 05:15:00 < 2010\text{-}02\text{-}05\ 05:30:00$).

---

## 2. Methodology & Formal Mathematical Definitions

All three strategies evaluate the identical set of candidate timestamps $\mathcal{C}_{\text{anom}} = \{t_1, t_2, \dots, t_{333}\}$ extracted from the multi-paradigm consensus pool. Let $\mathcal{T}_{\text{train}}$ denote the chronological training partition ($N_{\text{train}} = 109,347$ observations from Dec 16, 2006 to Feb 05, 2010), and let $P_t$ represent the observed Global Active Power in kW at test timestamp $t$.

### Strategy A: Global Historical Median (Context-Blind Baseline)
Strategy A collapses all diurnal, weekly, and seasonal dynamics into a single scalar population benchmark:
$$\mathcal{H}_t^{\text{Global}} = \mathcal{T}_{\text{train}}$$
$$\mu_{1/2}^{\text{Global}} = \text{median}\left(\{P_k \mid k \in \mathcal{H}_t^{\text{Global}}\}\right) = 0.6548\text{ kW}$$
$$\bar{P}^{\text{Global}} = \text{mean}\left(\{P_k \mid k \in \mathcal{H}_t^{\text{Global}}\}\right) = 1.1105\text{ kW}, \quad \sigma^{\text{Global}} = 1.0255\text{ kW}$$
$$\Delta_t^{\text{Global}} = |P_t - \mu_{1/2}^{\text{Global}}|$$
$$Z_t^{\text{Global}} = \frac{P_t - \bar{P}^{\text{Global}}}{\max(\sigma^{\text{Global}}, 10^{-6})}$$
Cohort size is constant: $|\mathcal{H}_t^{\text{Global}}| = 109,347$.

### Strategy B: Hour-Based and Weekend-Based Baseline (Static Bins)
Strategy B replicates the 48-bin lookup table from `household_energy_anomaly_detection.py`, segmenting training history strictly by time-of-day ($h_t \in \{0, \dots, 23\}$) and weekend status ($w_t \in \{0, 1\}$):
$$\mathcal{H}_t^{\text{HW}} = \{k \in \mathcal{T}_{\text{train}} \mid h_k = h_t \text{ and } w_k = w_t\}$$
$$\mu_{1/2}^{\text{HW}} = \text{median}\left(\{P_k \mid k \in \mathcal{H}_t^{\text{HW}}\}\right)$$
$$\bar{P}^{\text{HW}} = \text{mean}\left(\{P_k \mid k \in \mathcal{H}_t^{\text{HW}}\}\right), \quad \sigma^{\text{HW}} = \text{std}\left(\{P_k \mid k \in \mathcal{H}_t^{\text{HW}}\}\right)$$
$$\Delta_t^{\text{HW}} = |P_t - \mu_{1/2}^{\text{HW}}|$$
$$Z_t^{\text{HW}} = \frac{P_t - \bar{P}^{\text{HW}}}{\max(\sigma^{\text{HW}}, 10^{-6})}$$
Cohort size varies between $N = 1,292$ (weekends) and $N = 3,268$ (weekdays), averaging $2,440.6$ observations.

### Strategy C: Dynamic Historical Twin Matching (Member 4 Multi-Dimensional Cohort)
Strategy C extracts a dynamic cohort $\mathcal{H}_t^{\text{Dynamic}}$ tailored to diurnal, weekly, and seasonal proximity:
$$\mathcal{H}_t^{\text{Dynamic}} = \{k \in \mathcal{T}_{\text{train}} \mid (h_k - h_t) \bmod 24 \in \{-1, 0, 1\}, \, w_k = w_t, \, (m_k - m_t) \bmod 12 \in \{-1, 0, 1\}\}$$
Fallback Rule:
$$\text{If } |\mathcal{H}_t^{\text{Dynamic}}| < 50, \quad \mathcal{H}_t^{\text{Dynamic}} = \{k \in \mathcal{T}_{\text{train}} \mid (h_k - h_t) \bmod 24 \in \{-1, 0, 1\}, \, w_k = w_t\}$$
Contextual Metrics:
$$\mu_{1/2}^{\text{Dynamic}} = \text{median}\left(\{P_k \mid k \in \mathcal{H}_t^{\text{Dynamic}}\}\right)$$
$$\bar{P}^{\text{Dynamic}} = \text{mean}\left(\{P_k \mid k \in \mathcal{H}_t^{\text{Dynamic}}\}\right), \quad \sigma^{\text{Dynamic}} = \text{std}\left(\{P_k \mid k \in \mathcal{H}_t^{\text{Dynamic}}\}\right)$$
$$\Delta_t^{\text{Dynamic}} = |P_t - \mu_{1/2}^{\text{Dynamic}}|$$
$$Z_t^{\text{Dynamic}} = \frac{P_t - \bar{P}^{\text{Dynamic}}}{\max(\sigma^{\text{Dynamic}}, 10^{-6})}$$
Cohort size ranges between $N = 888$ and $N = 2,608$ (median: $2,340$).

---

## 3. Experimental Results and Quantitative Comparisons

### 3.1 Overall Distribution of Contextual Deviations ($\Delta_t = |P_t - \mu_{1/2}|$ in kW)

The table below summarizes the distribution of absolute contextual deviations computed across all 333 candidate anomaly timestamps:

| Statistic | Strategy A (Global Median) | Strategy B (Hour-Weekend) | Strategy C (Dynamic Twins) | Differential (C vs A) | Differential (C vs B) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean** | $2.7412\text{ kW}$ | $2.2798\text{ kW}$ | **$2.2097\text{ kW}$** | **$-0.5315\text{ kW}$ ($-19.4\%$)** | **$-0.0701\text{ kW}$ ($-3.1\%$)** |
| **Standard Deviation** | $1.5821\text{ kW}$ | $1.3551\text{ kW}$ | **$1.3258\text{ kW}$** | $-0.2563\text{ kW}$ | $-0.0293\text{ kW}$ |
| **Minimum** | $0.0037\text{ kW}$ | $0.0079\text{ kW}$ | **$0.0004\text{ kW}$** | $-0.0033\text{ kW}$ | $-0.0075\text{ kW}$ |
| **25th Percentile** | $1.4576\text{ kW}$ | $1.1329\text{ kW}$ | **$1.0515\text{ kW}$** | $-0.4061\text{ kW}$ | $-0.0814\text{ kW}$ |
| **Median (50th %)** | $3.0725\text{ kW}$ | $2.4633\text{ kW}$ | **$2.4051\text{ kW}$** | **$-0.6674\text{ kW}$ ($-21.7\%$)** | **$-0.0582\text{ kW}$ ($-2.4\%$)** |
| **75th Percentile** | $3.8485\text{ kW}$ | $3.2389\text{ kW}$ | **$3.1997\text{ kW}$** | $-0.6488\text{ kW}$ | $-0.0392\text{ kW}$ |
| **Maximum** | $7.1131\text{ kW}$ | $6.1850\text{ kW}$ | **$5.8219\text{ kW}$** | **$-1.2912\text{ kW}$** | **$-0.3631\text{ kW}$** |

### 3.2 Reference Cohort Size Comparison ($N$)

| Metric | Strategy A (Global) | Strategy B (Hour-Weekend) | Strategy C (Dynamic Twins) |
| :--- | :---: | :---: | :---: |
| **Minimum Cohort Size ($N$)** | $109,347$ | $1,292$ | **$888$** |
| **25th Percentile ($N$)** | $109,347$ | $1,296$ | **$936$** |
| **Median Cohort Size ($N$)** | $109,347$ | $3,260$ | **$2,340$** |
| **Mean Cohort Size ($N$)** | $109,347$ | $2,440.6$ | **$1,805.6$** |
| **75th Percentile ($N$)** | $109,347$ | $3,263$ | **$2,376$** |
| **Maximum Cohort Size ($N$)** | $109,347$ | $3,268$ | **$2,608$** |
| **Fallback Triggered ($N < 50$)** | **0 / 333 ($0.0\%$)** | **0 / 333 ($0.0\%$)** | **0 / 333 ($0.0\%$)** |

### 3.3 Contextual Z-Score Comparison ($Z_{\text{context}}$)

| Strategy | Mean $Z$ | Std $Z$ | Median $Z$ | Min $Z$ | Max $Z$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy A (Global)** | $+2.1819$ | $1.6216$ | $+2.5519$ | $-0.9079$ | $+6.4921$ |
| **Strategy B (Hour-Weekend)** | $+1.9185$ | $1.6739$ | $+2.0887$ | $-1.4410$ | $+6.6524$ |
| **Strategy C (Dynamic Twins)** | $+1.9343$ | $1.7412$ | $+2.0428$ | $-1.5725$ | **$+9.5650$** |

*Interpretation of Z-Scores:* Strategy C reaches a maximum $Z$-score of $+9.5650$ (compared to $+6.6524$ in Strategy B) on tightly conditioned summer intervals where baseline variance is minimal ($\sigma \approx 0.35\text{ kW}$), proving that seasonal twin matching provides sharper signal-to-noise resolution during low-volatility seasonal regimes.

---

## 4. Segmented Operational Comparisons

### 4.1 Nighttime vs. Daytime Operational Regimes
Candidate timestamps were partitioned into **Nighttime** ($22:00\text{ to }06:00$, hours $\in \{22, 23, 0, 1, 2, 3, 4, 5, 6\}$, $N = 47$) and **Daytime** ($07:00\text{ to }21:00$, $N = 286$):

| Segment | Strategy A Mean Deviation | Strategy B Mean Deviation | Strategy C Mean Deviation | Relative Shift (C vs A) | Relative Shift (C vs B) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Nighttime ($N = 47$)** | $2.0398\text{ kW}$ | $2.0009\text{ kW}$ | **$1.9084\text{ kW}$** | $-0.1314\text{ kW}$ ($-6.4\%$) | $-0.0925\text{ kW}$ ($-4.6\%$) |
| **Daytime ($N = 286$)** | $2.8564\text{ kW}$ | $2.3257\text{ kW}$ | **$2.2592\text{ kW}$** | **$-0.5972\text{ kW}$ ($-20.9\%$)** | **$-0.0665\text{ kW}$ ($-2.9\%$)** |

*Key Takeaway:*
- During daytime hours, domestic load varies dramatically. Strategy A inflates daytime deviations ($2.856\text{ kW}$) because it compares active consumption against the global baseload ($0.655\text{ kW}$). Strategy C accommodates daytime occupancy, reducing mean deviation to $2.259\text{ kW}$.
- During nighttime baseload periods, typical consumption drops below $0.35\text{ kW}$. Both Strategy B and Strategy C correctly adjust to off-peak baselines, preventing mischaracterization of small nighttime deviations.

### 4.2 Weekday vs. Weekend Behavioral Regimes
Candidates were partitioned into **Weekday** ($w_t = 0$, $N = 194$) and **Weekend** ($w_t = 1$, $N = 139$):

| Segment | Strategy A Mean Deviation | Strategy B Mean Deviation | Strategy C Mean Deviation | Strategy B Median $N$ | Strategy C Median $N$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Weekday ($N = 194$)** | $2.5784\text{ kW}$ | $2.2507\text{ kW}$ | **$2.1904\text{ kW}$** | $3,260$ | $2,340$ |
| **Weekend ($N = 139$)** | $2.9683\text{ kW}$ | $2.3204\text{ kW}$ | **$2.2366\text{ kW}$** | $1,296$ | $936$ |

*Key Takeaway:*
- Strategy A exhibits a large divergence between weekdays ($2.578\text{ kW}$) and weekends ($2.968\text{ kW}$) because it fails to capture elevated weekend domestic occupancy.
- Strategy C maintains consistent mean contextual deviations across both regimes ($2.190\text{ kW}$ weekday vs. $2.237\text{ kW}$ weekend, a difference of only $0.047\text{ kW}$), demonstrating balanced normalization across calendar occupancy regimes.

---

## 5. Divergence Analysis: Substantially Different Contextual Interpretations

To investigate where strategy choice fundamentally alters the diagnostic interpretation of candidate anomalies, we identified the cases with the greatest divergence between Strategy C (Dynamic Twin) and Strategy B (Static Hour-Weekend).

### Case Group 1: Winter Evening Baseline Elevation ($\mu_{1/2}^{\text{Dynamic}} \gg \mu_{1/2}^{\text{HW}}$)
During cold winter evenings (February), heating systems and early darkness create an elevated domestic baseline. Strategy B, which averages across the entire calendar year (including mild spring and warm summer days), significantly under-estimates the typical winter baseline:

| Timestamp | Actual Power ($P_t$) | Hour | Month | Static HW Median | Dynamic Twin Median | $\Delta_{\text{median}}$ (C $-$ B) | HW Deviation | Dynamic Deviation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2010-02-06 18:45:00** | **$6.403\text{ kW}$** | 18 | 2 (Feb) | $1.583\text{ kW}$ | **$2.460\text{ kW}$** | **$+0.877\text{ kW}$** | $4.820\text{ kW}$ | **$3.943\text{ kW}$** |
| **2010-02-07 18:45:00** | **$3.953\text{ kW}$** | 18 | 2 (Feb) | $1.583\text{ kW}$ | **$2.460\text{ kW}$** | **$+0.877\text{ kW}$** | $2.370\text{ kW}$ | **$1.493\text{ kW}$** |
| **2010-02-20 18:00:00** | **$3.793\text{ kW}$** | 18 | 2 (Feb) | $1.583\text{ kW}$ | **$2.460\text{ kW}$** | **$+0.877\text{ kW}$** | $2.210\text{ kW}$ | **$1.333\text{ kW}$** |
| **2010-02-20 18:15:00** | **$2.917\text{ kW}$** | 18 | 2 (Feb) | $1.583\text{ kW}$ | **$2.460\text{ kW}$** | **$+0.877\text{ kW}$** | $1.334\text{ kW}$ | **$0.457\text{ kW}$** |
| **2010-02-06 19:30:00** | **$3.413\text{ kW}$** | 19 | 2 (Feb) | $1.985\text{ kW}$ | **$2.800\text{ kW}$** | **$+0.815\text{ kW}$** | $1.428\text{ kW}$ | **$0.613\text{ kW}$** |

*Operational Diagnostic Impact:*
On `2010-02-20 18:15:00`, active power was $2.917\text{ kW}$.
- Under Strategy B, this is diagnosed as a substantial **$+1.334\text{ kW}$ deviation** above typical consumption ($1.84\times$ median), implying an anomalous surge.
- Under Strategy C, because typical February weekend evening consumption is already $2.460\text{ kW}$, the observed load represents only a mild **$+0.457\text{ kW}$ deviation** ($1.19\times$ median). Strategy C avoids overstating anomaly severity during high-consumption seasonal regimes.

### Case Group 2: Late Summer / Early Autumn Baseload Depression ($\mu_{1/2}^{\text{Dynamic}} \ll \mu_{1/2}^{\text{HW}}$)
Conversely, during September and June, longer daylight and absent heating significantly depress typical evening power. Strategy B dilutes this seasonal reduction by blending in high-load winter records:

| Timestamp | Actual Power ($P_t$) | Hour | Month | Static HW Median | Dynamic Twin Median | $\Delta_{\text{median}}$ (C $-$ B) | HW Deviation | Dynamic Deviation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2010-09-28 19:00:00** | **$2.472\text{ kW}$** | 19 | 9 (Sep) | $1.510\text{ kW}$ | **$0.805\text{ kW}$** | **$-0.705\text{ kW}$** | $0.962\text{ kW}$ | **$1.667\text{ kW}$** |
| **2010-09-09 19:30:00** | **$2.638\text{ kW}$** | 19 | 9 (Sep) | $1.510\text{ kW}$ | **$0.805\text{ kW}$** | **$-0.705\text{ kW}$** | $1.128\text{ kW}$ | **$1.833\text{ kW}$** |
| **2010-09-17 19:00:00** | **$4.749\text{ kW}$** | 19 | 9 (Sep) | $1.510\text{ kW}$ | **$0.805\text{ kW}$** | **$-0.705\text{ kW}$** | $3.239\text{ kW}$ | **$3.944\text{ kW}$** |
| **2010-06-08 19:45:00** | **$0.730\text{ kW}$** | 19 | 6 (Jun) | $1.510\text{ kW}$ | **$0.814\text{ kW}$** | **$-0.696\text{ kW}$** | $0.780\text{ kW}$ | **$0.084\text{ kW}$** |
| **2010-06-21 19:00:00** | **$3.128\text{ kW}$** | 19 | 6 (Jun) | $1.510\text{ kW}$ | **$0.814\text{ kW}$** | **$-0.696\text{ kW}$** | $1.618\text{ kW}$ | **$2.314\text{ kW}$** |

*Operational Diagnostic Impact:*
On `2010-09-28 19:00:00`, active power was $2.472\text{ kW}$.
- Under Strategy B, the load appears as a modest **$+0.962\text{ kW}$ deviation** above the annual median of $1.510\text{ kW}$.
- Under Strategy C, because typical September consumption at 19:00 is only $0.805\text{ kW}$, the observed load represents a major **$+1.667\text{ kW}$ surge** ($3.07\times$ typical seasonal load). Strategy C successfully uncovers acute deviations that are masked by annual averaging.

### Case Group 3: The Four-Model Showcase Anomaly (`2010-04-11 20:45:00`)
Evaluation of the canonical four-model consensus event across all three strategies:

| Metric | Strategy A (Global) | Strategy B (Hour-Weekend) | Strategy C (Dynamic Twin) |
| :--- | :---: | :---: | :---: |
| **Reference Cohort Size ($N$)** | $109,347$ | $1,296$ | **$936$** |
| **Contextual Median ($\mu_{1/2}$)** | $0.6548\text{ kW}$ | $1.9821\text{ kW}$ | **$1.9922\text{ kW}$** |
| **Contextual Mean ($\bar{P}$)** | $1.1105\text{ kW}$ | $1.9745\text{ kW}$ | **$2.0084\text{ kW}$** |
| **Contextual Std Dev ($\sigma$)** | $1.0255\text{ kW}$ | $1.5401\text{ kW}$ | **$1.4361\text{ kW}$** |
| **Absolute Deviation ($\Delta$)** | **$6.4333\text{ kW}$** | **$5.1060\text{ kW}$** | **$5.0959\text{ kW}$** |
| **Contextual Z-Score ($Z$)** | **$+5.83$** | **$+3.32$** | **$+3.55$** |
| **Consumption Ratio ($P_t / \mu_{1/2}$)** | **$10.82\times$** | **$3.58\times$** | **$3.56\times$** |

*Diagnostic Interpretation:*
- In April (spring), the annual Sunday evening median ($1.982\text{ kW}$) and the seasonal twin median ($1.992\text{ kW}$) converge, yielding nearly identical deviation metrics ($\approx 5.10\text{ kW}$).
- However, Strategy A dramatically distorts the diagnosis, reporting a $10.8\times$ surge relative to the global population baseline ($0.655\text{ kW}$), obscuring the fact that Sunday evenings naturally exhibit higher baseline demand.

---

## 6. Implementation Issues & Scientific Guardrail Audit

During execution of this experiment, the following implementation characteristics and guardrails were verified:

1. **Temporal Leakage Audit:**
   - Script asserted: `assert max_twin_timestamps <= max_train_ts`.
   - Verified that all historical reference observations across all 333 candidates had timestamps $\le 2010\text{-}02\text{-}05\ 05:15:00$.
   - **Verdict: ZERO LOOKAHEAD LEAKAGE.**
2. **Missing Telemetry & Cohort Insufficiency:**
   - Zero missing active power values were detected in the 333 candidates (`is_power_missing = False` for all rows).
   - Zero cohorts had fewer than 50 observations (`is_cohort_insufficient = False` for all rows).
3. **No Overwriting of Baseline Assets:**
   - Existing production files (`Anomaly-Detection-main/outputs/contextual_anomalies.csv`, `explanations.json`) remained untouched.
   - All Phase 3 outputs were written strictly to `experiments/phase3_context_matching/`.
4. **Academic Integrity Wording:**
   - In accordance with project instructions, no alert is classified as a "false positive" or "true positive" because the UCI benchmark dataset lacks ground-truth fault logs.
   - The report avoids claiming that Strategy C is "statistically superior"; rather, the analysis demonstrates that Strategy C provides greater seasonal and diurnal specificity, reducing baseline distortion during transitional seasonal regimes.

---

## 7. Deliverables & Artifact Manifest

All generated datasets, scripts, and logs are located in `experiments/phase3_context_matching/`:

| Artifact | File Path | Records / Shape | Description |
| :--- | :--- | :---: | :--- |
| **Strategy A CSV** | [`strategy_a_global.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_context_matching/strategy_a_global.csv) | 333 rows $\times$ 10 cols | Global historical baseline evaluation for 333 candidates. |
| **Strategy B CSV** | [`strategy_b_hour_weekend.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_context_matching/strategy_b_hour_weekend.csv) | 333 rows $\times$ 10 cols | 48-bin static hour $\times$ weekend baseline evaluation. |
| **Strategy C CSV** | [`strategy_c_dynamic_twins.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_context_matching/strategy_c_dynamic_twins.csv) | 333 rows $\times$ 10 cols | Dynamic historical twin matching evaluation. |
| **Combined CSV** | [`combined_strategy_comparison.csv`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_context_matching/combined_strategy_comparison.csv) | 333 rows $\times$ 28 cols | Unified side-by-side dataset with differential metrics. |
| **Python Script** | [`run_phase3_experiment.py`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_context_matching/run_phase3_experiment.py) | 320 lines | Fully reproducible, leak-free execution script. |
| **Markdown Report** | [`phase3_context_matching_report.md`](file:///c:/Users/hp/Downloads/Anomaly-Detection-main/experiments/phase3_context_matching/phase3_context_matching_report.md) | Markdown document | Complete experimental analysis, case studies, and findings. |

---

## 8. Conclusion

Phase 3 Experiment 1 establishes clear empirical evidence regarding how contextual baseline design governs anomaly interpretation:
1. **Global baselines (Strategy A)** fail to differentiate peak domestic activity from baseline consumption, inflating daytime deviations by over $20\%$.
2. **Static hour-weekend baselines (Strategy B)** capture time-of-day dynamics, but fail to account for seasonal heating and cooling cycles, under-representing winter evening baselines by up to $+0.88\text{ kW}$ and over-representing summer evening baselines by up to $+0.70\text{ kW}$.
3. **Dynamic Historical Twin Matching (Strategy C)** provides robust, leak-free contextualization that balances seasonal climate shifts and diurnal habits while preserving large, statistically valid reference cohorts ($N \ge 888$, median $2,340$).

