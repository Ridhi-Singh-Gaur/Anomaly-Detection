# Household Electricity Anomaly Detection Using Machine Learning and LSTM

## 1. About the Project

Household Electricity Anomaly Detection is a machine learning project that identifies unusual patterns in household electricity consumption.

Electricity consumption changes throughout the day. For example, consumption may increase when a washing machine, air conditioner, or water heater is being used. However, an unexpected spike or sudden change may deserve further investigation.

The purpose of this project is to learn normal consumption patterns and identify readings that differ significantly from those patterns.

The project uses four methods:

1. **Statistical Baseline:** Identifies consumption that differs significantly from typical consumption for a particular time of day.
2. **Isolation Forest:** Detects unusual combinations of electricity measurements and time-related features.
3. **LSTM Prediction:** Predicts the next electricity reading and identifies large differences between actual and predicted values.
4. **LSTM Autoencoder:** Learns to reconstruct common consumption patterns and flags sequences with unusually large reconstruction errors.

The pipeline automatically performs data cleaning, feature engineering, model training, anomaly detection, evaluation, and result generation.

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
- **Sub_metering_1:** Energy consumed by the kitchen sub-meter, including appliances such as the dishwasher, oven, and microwave. Measured in watt-hours (Wh).
- **Sub_metering_2:** Energy consumed by the laundry-room sub-meter, including appliances such as the washing machine, tumble dryer, and refrigerator. Measured in watt-hours (Wh).
- **Sub_metering_3:** Energy consumed by the electric water heater and air conditioner. Measured in watt-hours (Wh).

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

## 9. Generated Output Files

All results are saved in the `outputs/` directory.

### Processed Data and Results

- **`household_15min_features.csv`:** Cleaned 15-minute dataset and engineered features.
- **`statistical_baseline_results.csv`:** Expected consumption, deviations, thresholds, and anomaly flags.
- **`isolation_forest_results.csv`:** Isolation Forest scores and anomaly flags.
- **`lstm_prediction_results.csv`:** Actual and predicted power, prediction errors, and flags.
- **`lstm_autoencoder_results.csv`:** Reconstruction errors and flags.
- **`classical_model_comparison.csv`:** Comparison of the statistical baseline and Isolation Forest.
- **`deep_learning_model_comparison.csv`:** Comparison of the two LSTM methods.
- **`model_evaluation_metrics.csv`:** MAE and RMSE values.
- **`model_metadata.json`:** Model settings and anomaly thresholds.

### Saved Models

- **`isolation_forest_model.joblib`:** Saved Isolation Forest model.
- **`power_scaler.joblib`:** Fitted power-scaling object.
- **`lstm_prediction_model.keras`:** Saved LSTM predictor.
- **`lstm_autoencoder_model.keras`:** Saved LSTM autoencoder.

### Generated Graphs

- **`consumption_first_week.png`:** Consumption changes over the first week of available data.
- **`average_consumption_by_hour.png`:** Average consumption for each hour of the day.
- **`classical_model_comparison.png`:** Flagged percentages for the classical models.
- **`lstm_prediction_error_distribution.png`:** Distribution of LSTM prediction errors.
- **`deep_learning_model_comparison.png`:** Flagged percentages for the LSTM models.
- **`lstm_prediction_plot.png`:** Actual versus predicted consumption, with flagged readings highlighted.

The CSV files allow individual candidate events to be inspected. Where a results file contains anomaly scores, sorting by score helps identify the most unusual readings according to that model.

## 10. Technologies Used

- **Python:** Overall implementation.
- **Pandas and NumPy:** Data processing, feature engineering, and numerical calculations.
- **Scikit-learn:** Isolation Forest, scaling, and regression metrics.
- **TensorFlow/Keras:** LSTM prediction and autoencoder models.
- **Matplotlib:** Graph generation.
- **Joblib:** Saving trained models and preprocessing objects.

## 11. How to Run the Project

### Install Dependencies

```
pip install pandas numpy matplotlib scikit-learn tensorflow joblib
```

### Download the Dataset

Download the dataset from the [official UCI repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) and place the text file in your project directory.

### Run the Pipeline

If your downloaded file is named `household_power_consumption.txt`:

```
python .\household_energy_anomaly_detection.py --input ".\household_power_consumption.txt"
```

If your local file is still named `household_power_consumption (1).txt`, use:

```
python .\household_energy_anomaly_detection.py --input ".\household_power_consumption (1).txt"
```

The script creates the `outputs/` directory if necessary, preprocesses the data, trains the models, calculates evaluation metrics, and saves results and graphs.

TensorFlow CPU support is sufficient for the configuration used in your successful run. A GPU warning does not necessarily indicate a training failure.

## 12. Conclusion

This project implements an end-to-end household electricity anomaly-detection pipeline using statistical methods, Isolation Forest, and two LSTM-based approaches.

It processes the original UCI dataset into **136,684 15-minute observations** and evaluates the models on **27,337 later observations**. It then saves candidate anomaly results, evaluation metrics, trained models, and visualisations.

The main objective is to identify electricity readings that deserve further investigation. Future improvements can include validating flagged events, improving feature construction for real-time prediction, testing longer historical sequences, and evaluating the models against verified anomaly labels.

**The current results confirm successful pipeline execution, but further validation is required to establish anomaly-detection accuracy.**

## References

1. UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
2. Liu, Ting and Zhou, *Isolation Forest*, 2008.
3. Hochreiter and Schmidhuber, *Long Short-Term Memory*, 1997.
4. TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
5. Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)

