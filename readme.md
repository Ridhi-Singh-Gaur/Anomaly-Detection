# Context-Aware Explainable AI for Household Electricity Anomaly Detection

## 1. About the Project

This repository implements an end-to-end household electricity anomaly detection and **Context-Aware Explainable AI (XAI)** pipeline using statistical profiling, classical machine learning, deep recurrent sequence models, and game-theoretic feature attribution.

Electricity consumption varies continuously with diurnal schedules, weekend routines, and seasonal heating/cooling demand. Rather than issuing opaque binary flags, this project combines multi-paradigm detection with leak-free historical context matching and TreeSHAP path-length inversion to produce verifiable, evidence-bounded explanations across **333 candidate anomalies** identified in held-out test data.

The project evaluates four detection models:

1. **Statistical Baseline:** Identifies consumption that differs significantly from typical median consumption for the same `(hour, is_weekend)` regime.
2. **Isolation Forest:** Detects unusual combinations of 16 multivariate electricity measurements, temporal flags, and rolling/lag features.
3. **LSTM Prediction:** Supervised continuous next-step regressor trained on 4-step power windows to flag large prediction residuals.
4. **LSTM Autoencoder:** Unsupervised 4-step sequence encoder-decoder with an overcomplete 32-dimensional latent state that flags sequences with large trajectory reconstruction MSE.

Across **27,337 chronological test intervals**, the four models collectively surface **333 candidate timestamps** (`1.218%`). On top of those frozen predictions, the XAI layer adds:

- **Historical Twin Contextualization (`context_analysis.py`):** Compares each candidate anomaly against training-only reference cohorts matching `±1 h` (modulo 24), identical weekend status, and `±1 month` (modulo 12).
- **TreeSHAP Path-Length Inversion (`shap_explainer.py`):** Decomposes Isolation Forest average path length h(x) into additive anomaly-severity attributions (phi_anom = -phi_path).
- **Evidence-Bounded Diagnostics (`explanation_generator.py` & `context_aware_xai.py`):** Synthesizes multi-model consensus, contextual deviations, sub-metering clues, and directionally categorized SHAP features without speculative appliance or causal claims.

**Important:** The models identify candidate anomalies, not confirmed electrical faults. High electricity consumption may be normal depending on household activities.

## 2. Dataset Description

### Dataset Source

This project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

**Dataset link:** [https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

The dataset contains approximately 2 million minute-level electricity measurements collected from a single household between December 2006 and November 2010.

### How to Download the Dataset

1. Open the [official UCI dataset page](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
2. Download the dataset file or its archive.
3. Extract the archive if necessary.
4. Place the text file in your project directory.

The original dataset file is named `household_power_consumption.txt`.

The raw dataset is not included in this repository because it exceeds GitHub's 100 MB file-size limit.

### Understanding the Original Dataset Columns

The original dataset contains nine columns: two time-related columns and seven electrical measurements.

- **Date:** The date on which the electricity reading was recorded.
- **Time:** The time at which the reading was recorded.
- **Global_active_power:** Total active power consumed by the household, measured in kilowatts (kW). This is the main variable used for anomaly detection.
- **Global_reactive_power:** Reactive power associated with household electrical equipment, measured in kilowatts (kW).
- **Voltage:** Voltage supplied to the household, measured in volts (V).
- **Global_intensity:** Electrical current drawn by the household, measured in amperes (A).
- **Sub_metering_1:** Energy consumed per minute by the kitchen sub-meter (dishwasher, oven, microwave in UCI metadata), recorded in watt-hours (Wh) at 1-minute resolution (`Wh/min` average rate after 15-minute `.mean()` resampling, corresponding to `P_avg = 60 × rate` W or `15 × rate` Wh cumulative per 15-minute slot).
- **Sub_metering_2:** Energy consumed per minute by the laundry-room sub-meter (washing machine, tumble dryer, refrigerator, light in UCI metadata), recorded in Wh (`Wh/min` average rate after 15-minute resampling).
- **Sub_metering_3:** Energy consumed per minute by the electric water heater and air conditioner sub-meter in UCI metadata, recorded in Wh (`Wh/min` average rate after 15-minute resampling).

**Understanding the main variable:** `Global_active_power` represents the household's overall active power consumption. A sudden increase may indicate unusual usage, but it is not automatically an anomaly.

The dataset does not contain verified labels for actual electrical faults. Therefore, the project detects unusual patterns rather than calculating accuracy against known fault labels.

## 3. Step 1: Data Cleaning and Preprocessing

The original dataset contains minute-level observations, missing values, and separate date and time columns. Before training the models, the project prepares the data.

### What Happens During Preprocessing?

1. **Read the dataset:** Load the original semicolon-separated text file using Pandas.
2. **Create timestamps:** Combine `Date` and `Time` into a single timestamp.
3. **Clean the data:** Convert electrical measurements to numeric values, remove invalid timestamps, sort observations chronologically, and remove duplicate timestamps.
4. **Resample to 15-minute intervals:** Calculate the mean of available measurements within each interval. This reduces the dataset size while preserving useful consumption patterns.
5. **Handle missing values:** Interpolate short gaps of up to four consecutive 15-minute intervals when surrounded by existing observations. Remove rows without valid active-power readings.
6. **Save processed data:** Store the cleaned dataset and engineered features in `outputs/household_15min_features.csv`.

### Results of Preprocessing

Your execution produced these results:

- Raw rows reported after timestamp cleaning and duplicate removal: **2,075,259**
- Processed 15-minute observations: **136,684**

The dataset becomes smaller because multiple minute-level readings are combined into one 15-minute average, and rows without valid active-power values are removed.

## 4. Step 2: Feature Engineering

Feature engineering creates additional columns that help the models understand when electricity is consumed and how consumption changes over time.

The script generates the following features:

- **timestamp:** Combined date and time of each observation.
- **hour:** Hour of the day, from 0 to 23. Helps identify daily consumption patterns.
- **day_of_week:** Day of the week, where Monday = 0 and Sunday = 6.
- **is_weekend:** 1 for Saturday and Sunday; otherwise 0.
- **month:** Month of the year, from 1 to 12.
- **day:** Day of the month, from 1 to 31.
- **date:** Date stored as a string for easier grouping and inspection.
- **lag_1:** Active power from the previous 15-minute interval.
- **lag_4:** Active power from four intervals earlier, approximately one hour earlier.
- **lag_96:** Active power from 96 intervals earlier, approximately 24 hours earlier.
- **rolling_mean_4:** Average active power over the current and previous three readings.
- **rolling_std_4:** Variation in active power across the four-reading window.
- **consumption_change:** Difference between the current active power and the previous reading.

For example, if consumption changes from 0.8 kW to 2.0 kW:

`2.0 - 0.8 = 1.2 kW`

This represents a sharp increase, although additional analysis is needed to determine whether it is unusual.

These features provide useful context for anomaly detection. The current Isolation Forest feature list does not include `lag_96`.

## 5. Step 3: Training and Testing

The processed dataset is sorted by timestamp and split chronologically:

- **80% training data:** Used to learn consumption patterns and determine anomaly thresholds.
- **20% testing data:** Used to evaluate the trained models on later observations.

Your run contains approximately:

- Training observations: **109,347**
- Testing observations: **27,337**

Chronological splitting is important because future observations should not be randomly mixed into the training data when evaluating time-series models.

## 6. Step 4: Anomaly Detection Models

### Model 1: Statistical Baseline

**What does it do?**

Identifies readings that differ significantly from the usual consumption for the same hour of the day and weekend status.

**How does it work?**

1. Calculate median active power for each hour/weekend combination using training data.
2. Compare each reading against its corresponding expected consumption.
3. Calculate the absolute difference between actual and expected consumption.
4. Set the anomaly threshold to the 99th percentile of training deviations.
5. Flag test readings whose deviations exceed the threshold.

**Example:** If expected consumption is 0.8 kW and actual consumption is 2.4 kW, the deviation is 1.6 kW. If it exceeds the learned threshold, the reading is flagged.

This method is simple and interpretable, but it may not capture complex consumption patterns.

### Model 2: Isolation Forest

**What does it do?**

Detects observations that are unusual compared with the overall distribution of electricity measurements and engineered features.

**How does it work?**

1. Handle missing feature values using medians calculated from training data.
2. Standardise features so their numerical scales are more comparable.
3. Train an Isolation Forest containing 200 trees.
4. Calculate anomaly scores for test observations.
5. Flag readings that exceed the learned training-score threshold.

Isolation Forest works by randomly partitioning observations. Unusual observations can often be isolated with fewer partitions than typical observations.

Unlike the statistical baseline, it considers multiple features together, such as active power, voltage, current, and recent consumption.

### Model 3: LSTM Next-Step Prediction

**What does it do?**

Learns to predict the next electricity reading using previous readings. A large prediction error may indicate an unusual consumption pattern.

**How does it work?**

1. Scale active-power values between 0 and 1 using a scaler fitted on training data.
2. Create sequences of four consecutive readings, representing approximately one hour.
3. Train an LSTM neural network to predict the next 15-minute reading.
4. Compare actual test readings with predicted readings.
5. Flag readings whose prediction errors exceed the 99th-percentile training-error threshold.

**Workflow:**

Previous readings → LSTM model → Predicted next reading → Compare with actual reading → Calculate prediction error → Flag unusually large errors.

LSTM is useful because it learns sequential patterns instead of relying only on fixed statistical rules.

### Model 4: LSTM Autoencoder

**What does it do?**

Learns to reconstruct common electricity consumption sequences. A sequence reconstructed poorly may be unusual compared with the patterns learned during training.

**How does it work?**

1. Create sequences of four consecutive active-power readings.
2. Pass each sequence through an LSTM encoder that learns a representation of the sequence.
3. Use an LSTM decoder to reconstruct the original sequence.
4. Calculate the mean squared reconstruction error.
5. Flag test sequences whose errors exceed the 99th-percentile training-error threshold.

**Difference between the two LSTM models:**

- **LSTM Prediction:** Predicts the next reading.
- **LSTM Autoencoder:** Reconstructs the input sequence it receives.

Both methods use error thresholds to identify candidate anomalies, but their errors measure different things.

## 7. Results from Your Execution

The pipeline completed all four methods and saved the output files.

### Classical Models

**Statistical Baseline**

- Test readings: 27,337
- Flagged readings: 82
- Flagged percentage: 0.300%

The model flagged readings that differed significantly from the learned hour/weekend consumption baseline.

**Isolation Forest**

- Test readings: 27,337
- Flagged readings: 93
- Flagged percentage: 0.340%

The model flagged readings that were unusual according to its multivariate anomaly scores.

### Deep Learning Models

**LSTM Prediction**

- Test readings: 27,337
- Flagged readings: 155
- Flagged percentage: 0.567%

The model flagged readings with unusually large prediction errors.

**LSTM Autoencoder**

- Test readings: 27,337
- Flagged readings: 139
- Flagged percentage: 0.508%

The model flagged sequences with unusually large reconstruction errors.

### How Should These Results Be Interpreted?

All four models successfully produced candidate anomaly flags.

The LSTM predictor flagged the largest number of readings in this run, but that does not mean it performed best. The models learn different patterns and use different error calculations.

Because the dataset has no verified anomaly labels, the true correctness of the flagged readings is unknown.

**These results demonstrate successful pipeline execution, not verified anomaly-classification accuracy.**

## 8. Understanding MAE and RMSE

Your script also reports the following test-set metrics.

### LSTM Prediction

- **MAE:** 0.299672 kW
- **RMSE:** 0.473670 kW

MAE is the average absolute difference between actual and predicted readings. RMSE gives greater weight to larger errors.

The predictor's MAE of approximately 0.300 kW describes its average next-step prediction error.

### LSTM Autoencoder

- **MAE:** 0.002561 kW
- **RMSE:** 0.004308 kW

These metrics describe reconstruction of the ending reading in each test sequence, rather than next-step prediction.

The autoencoder's very low errors should be investigated further. The metrics cannot be directly compared because the models perform different tasks. Low reconstruction error also does not establish that genuine anomalies are being detected correctly.

## 9. Context-Aware Explainable AI (XAI) Pipeline & Validation Suite

To make anomaly flags transparent and auditable, four XAI modules extend the base detector outputs without altering trained weights, predictions, thresholds, or timestamps:

1. **`context_analysis.py` — Historical Twin Matching:** Retrieves training-only reference cohorts (`t <= 2010-02-05 05:15:00`, `N_train = 109,347`) matching circular hour tolerance (`d_h <= 1`), identical `is_weekend` flag, and circular month proximity (`d_m <= 1`, with fallback to hour × weekend when cohort size `< 50`). Computes `context_expected_median_kw`, `context_abs_deviation_kw`, `context_percentile`, `context_z_score`, and `context_cohort_size`.
2. **`shap_explainer.py` — TreeSHAP Path-Length Inversion:** Runs `shap.TreeExplainer` over the 200-tree `IsolationForest` predicting average isolation path length `h(x)` (baseline `phi_0 = 11.949459` depth levels) and inverts sign (`phi_anom = -phi_path`) so positive values reflect features pushing towards anomaly severity. Local additivity holds within `<= 2.75e-14` relative to model predictions.
3. **`explanation_generator.py` — Deterministic Evidence-Bounded Diagnostics:** Categorizes SHAP features directionally (contributing towards severity `phi >= +0.05`, towards normality `phi <= -0.05`, or negligible `|phi| < 0.05`) and reports observed sub-circuit telemetry without inventing causal fault diagnoses or unmeasured appliance claims.
4. **`context_aware_xai.py` — Master XAI Pipeline:** Merges all four model results across the **333 candidate timestamps** (`242` single-model, `50` two-model, `37` three-model, `4` four-model consensus; `6` shared between LSTM Predictor and LSTM Autoencoder = `1.80%`) and generates the CSV, JSON, PNG, and Markdown deliverables below.

### Phase 3 Validation Experiments (`experiments/`)

| Folder | Experiment | Key Verified Finding |
| :--- | :--- | :--- |
| `experiments/phase3_context_matching/` | Reference Strategy Comparison | Dynamic Twins lower mean contextual deviation across the 333 candidates to **2.2097 kW** (vs. **2.7412 kW** global baseline, `-19.4%`; **2.2798 kW** static hour-weekend), median cohort **2,340**, `0/333` fallbacks. |
| `experiments/phase3_context_sensitivity/` | `3 × 2` Window & Seasonality Grid | Evaluated `h in {0, 1, 2} × Season in {Off, On}`; cohort sizes scale `1 : 3 : 5`, `sigma_min = 0.3363 kW`, preventing winter/summer baseline bias (`+0.962` / `-0.705 kW`). |
| `experiments/phase3_shap_stability/` | Deterministic Tree Subset Convergence | Nested subsets `K in {25, 50, 100, 150}` converge monotonically to the 200-tree reference (Spearman `rho`: `0.8706 -> 0.9971`, 100% Top-5 overlap at `K >= 100`). |
| `experiments/phase3_shap_randomized_stability/` | 20-Seed Monte Carlo TreeSHAP Check | Across `80` randomized draws (`20 × 4` sizes), `K = 150` achieves `rho = 0.9957 ± 0.0031` and **95.0%** exact Top-5 recovery with zero prefix-ordering bias (`<= 1.84e-14` additivity error). |
| `experiments/phase3_shap_dependence/` | Collinearity & Grouped Attribution | `Global_active_power` and `Global_intensity` correlate at `r = 0.9994`; post-hoc grouped attribution `phi_bulk = phi_P + phi_I` averages **1.0540 path-length units** (`+37.9%` above rank-1 `Sub_metering_2`). |
| `experiments/phase3_explanation_semantic_audit/` | Audit & Partition Reconciliation | Verified **333/333** candidates strictly in test partition (`0` in train, zero lookahead leakage) and surfaced `78` baseline template warnings across `73` events. |
| `experiments/phase3_explanation_remediation/` | Explanation Quality Remediation | Resolved all `78` wording warnings (directional SHAP headings + evidence-bounded residual load phrasing) achieving **333/333 PASS (`0 WARNING`, `0 FAIL`)**. |
| `experiments/phase3_final_integration/` | Cross-Experiment Integration Audit | Confirmed `<= 8.88e-16` tabular float parity against `outputs_backup_phase3/`, frozen artifact integrity, and full test pass coverage. |

## 10. Generated Output Files

All outputs are stored in `outputs/` (with pre-remediation baselines preserved in `outputs_backup_phase3/`):

### Processed Data, Model Results & XAI Records

- **`household_15min_features.csv`:** Cleaned 15-minute dataset and 16+ engineered features (`136,684` rows).
- **`statistical_baseline_results.csv`**, **`isolation_forest_results.csv`**, **`lstm_prediction_results.csv`**, **`lstm_autoencoder_results.csv`:** Test-partition anomaly scores, thresholds, and boolean flags (`27,337` rows each).
- **`classical_model_comparison.csv`**, **`deep_learning_model_comparison.csv`**, **`model_evaluation_metrics.csv`**, **`model_metadata.json`:** Model summaries, MAE/RMSE metrics, and saved cutoffs.
- **`contextual_anomalies.csv`:** Master table across all `333` candidate events including model flags, consensus count, historical twin cohort size/median/deviation/percentile/Z-score, sub-metering readings, top SHAP features, and audited diagnostic summaries.
- **`explanations.json`:** Structured JSON records for all `333` candidate anomalies.
- **`xai_execution_report.md`:** Summary report of candidate census, global TreeSHAP feature rankings, and the showcase four-way consensus case study (`2010-04-11 20:45:00`).

### Saved Models

- **`isolation_forest_model.joblib`:** Saved 200-tree Isolation Forest model.
- **`power_scaler.joblib`:** Fitted MinMax scaler object.
- **`lstm_prediction_model.keras`:** Saved Next-Step LSTM Regressor.
- **`lstm_autoencoder_model.keras`:** Saved LSTM Sequence Autoencoder.

### Generated Visualizations

- **`consumption_first_week.png`:** Consumption changes over the first week of available data.
- **`average_consumption_by_hour.png`:** Average consumption for each hour of the day.
- **`classical_model_comparison.png`:** Flagged percentages for the classical models.
- **`lstm_prediction_error_distribution.png`:** Distribution of LSTM prediction errors.
- **`deep_learning_model_comparison.png`:** Flagged percentages for the LSTM models.
- **`lstm_prediction_plot.png`:** Actual versus predicted consumption, with flagged readings highlighted.
- **`shap_summary.png`:** Global mean absolute TreeSHAP feature importance across the `333` candidates (`Sub_metering_2 = 0.7644`, `rolling_std_4 = 0.6584`, `Global_intensity = 0.5600`).
- **`individual_anomaly_explanation.png`:** Local TreeSHAP severity attribution profile for showcase consensus event `2010-04-11 20:45:00`.

## 11. Repository Structure

```text
├── household_energy_anomaly_detection.py   # End-to-end preprocessing, training & baseline evaluation
├── context_analysis.py                     # Leak-free Historical Twin cohort matching
├── shap_explainer.py                       # TreeSHAP isolation path-length inversion
├── explanation_generator.py                # Directionally categorized, evidence-bounded NLG
├── context_aware_xai.py                    # Master XAI pipeline orchestrator
├── outputs/                                # Models, test results, XAI CSV/JSON & figures
├── outputs_backup_phase3/                  # Preserved pre-remediation baseline outputs
├── experiments/                            # Phase 3 validation experiments (scripts, logs, CSVs, reports)
└── tests/                                  # Automated pytest regression & verification suite
```

## 12. Technologies Used

- **Python 3.12:** Core implementation and orchestration.
- **Pandas, NumPy & SciPy:** Time-series resampling, feature engineering, contextual cohort statistics, and rank correlations.
- **Scikit-learn & Joblib:** Isolation Forest modeling, feature standardization/scaling, and model serialization.
- **TensorFlow / Keras:** LSTM predictor and sequence autoencoder architectures.
- **SHAP (`TreeExplainer`):** Exact tree path-length game-theoretic feature attributions.
- **Matplotlib & Pytest:** Diagnostic visualizations and automated regression verification.

## 13. How to Run and Test the Project

### Install Dependencies

```bash
pip install pandas numpy scipy matplotlib scikit-learn tensorflow joblib shap pytest
```

### Download the Dataset & Run Base Models *(Optional — Precomputed Outputs Included)*

Download `household_power_consumption.txt` from the [official UCI repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), place it in the project root, and run:

```bash
python .\household_energy_anomaly_detection.py --input ".\household_power_consumption.txt"
```

### Run the Context-Aware XAI Pipeline

Using the serialized models and CSVs already stored in `outputs/`:

```bash
python .\context_aware_xai.py
```

This regenerates `contextual_anomalies.csv`, `explanations.json`, `shap_summary.png`, `individual_anomaly_explanation.png`, and `xai_execution_report.md` inside `outputs/`.

### Run Automated Verification Tests

```bash
python -m pytest -v tests/
```

All `54` functional, mathematical additivity, cohort-leakage, collinearity, and semantic-audit tests pass against the repository outputs (`7` manuscript-freeze checks skip automatically when unpublished manuscript drafts are excluded).

## 14. Conclusion

This project delivers a reproducible household electricity anomaly detection and Context-Aware Explainable AI framework over **136,684 uniform 15-minute intervals** (`109,347` training / `27,337` test). Integrating four diverse detectors (`333` candidate alerts) with training-only Historical Twin Matching, mathematically verified TreeSHAP path-length inversion, collinear bulk attribution grouping, and semantic quality checks converts uncalibrated anomaly scores into transparent, evidence-bounded diagnostics without overfitting or lookahead leakage.

**Note:** Because the underlying UCI dataset contains no verified fault labels, flagged events and explanations indicate statistical and contextual anomalies deserving investigation rather than confirmed appliance defects.

## References

1. UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
2. Liu, F.T., Ting, K.M., Zhou, Z.-H.: *Isolation Forest*, IEEE ICDM (2008).
3. Hochreiter, S., Schmidhuber, J.: *Long Short-Term Memory*, Neural Computation (1997).
4. Lundberg, S.M. et al.: *From local explanations to global understanding with explainable AI for trees*, Nature Machine Intelligence (2020).
5. TensorFlow & Scikit-learn documentation: [tensorflow.org](https://www.tensorflow.org/), [scikit-learn.org](https://scikit-learn.org/)

