"""
HOUSEHOLD ELECTRICITY ANOMALY DETECTION — COMPLETE PIPELINE
Starts from the original UCI raw text dataset, prepares the data, then runs
statistical baseline, Isolation Forest, LSTM prediction, and LSTM autoencoder.

Run:
  pip install pandas numpy matplotlib scikit-learn tensorflow joblib
  python household_energy_anomaly_detection.py --input household_power_consumption.txt

Outputs (processed dataset, graphs, model files and results) are written to ./outputs.
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


# ===============================================
# DATA PREPARATION — CLEANING, RESAMPLING, FEATURES, EDA
# ===============================================

def prepare_electricity_data(raw_file: str, output_dir: Path) -> pd.DataFrame:
    """Read the original UCI semicolon-separated TXT and create 15-minute features."""
    print("\n" + "=" * 68)
    print("DATA PREPARATION: CLEANING, RESAMPLING AND FEATURE ENGINEERING")
    print("=" * 68)

    raw = pd.read_csv(
        raw_file,
        sep=";",
        na_values=["?", ""],
        low_memory=False,
    )
    raw.columns = [c.strip() for c in raw.columns]

    required = {"Date", "Time", "Global_active_power"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(
            f"Raw dataset is missing columns {sorted(missing)}. "
            "Download and extract household_power_consumption.txt from UCI."
        )

    raw["timestamp"] = pd.to_datetime(
        raw["Date"].astype(str).str.strip() + " " + raw["Time"].astype(str).str.strip(),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    numeric_columns = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    for column in numeric_columns:
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")
    raw = raw.drop_duplicates(subset=["timestamp"]).set_index("timestamp")

    available_numeric = [c for c in numeric_columns if c in raw.columns]
    minute_data = raw[available_numeric].resample("15min").mean()
    # Interpolate short gaps only; long gaps remain missing and are dropped below.
    minute_data[available_numeric] = minute_data[available_numeric].interpolate(
        method="time", limit=4, limit_area="inside"
    )
    minute_data = minute_data.dropna(subset=["Global_active_power"])
    data = minute_data.reset_index()

    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["month"] = data["timestamp"].dt.month
    data["day"] = data["timestamp"].dt.day
    data["date"] = data["timestamp"].dt.date.astype(str)
    data["lag_1"] = data["Global_active_power"].shift(1)
    data["lag_4"] = data["Global_active_power"].shift(4)
    data["lag_96"] = data["Global_active_power"].shift(96)
    data["rolling_mean_4"] = data["Global_active_power"].rolling(4, min_periods=1).mean()
    data["rolling_std_4"] = data["Global_active_power"].rolling(4, min_periods=2).std()
    data["consumption_change"] = data["Global_active_power"].diff()

    # Keep rows with valid target consumption. Remaining feature NaNs are handled
    # using training-only imputation in the classical model.
    data = data.dropna(subset=["Global_active_power"]).reset_index(drop=True)
    processed_path = output_dir / "household_15min_features.csv"
    data.to_csv(processed_path, index=False)

    # Basic EDA graphs.
    plt.figure(figsize=(12, 4))
    sample = data.iloc[: min(len(data), 7 * 96)]
    plt.plot(sample["timestamp"], sample["Global_active_power"], linewidth=0.8)
    plt.title("Household active power — first week of available 15-minute data")
    plt.xlabel("Timestamp")
    plt.ylabel("Active power (kW)")
    plt.tight_layout()
    plt.savefig(output_dir / "consumption_first_week.png", dpi=150)
    plt.close()

    hourly = data.groupby("hour")["Global_active_power"].mean()
    plt.figure(figsize=(9, 4))
    hourly.plot(kind="bar")
    plt.title("Average active power by hour of day")
    plt.xlabel("Hour of day")
    plt.ylabel("Mean active power (kW)")
    plt.tight_layout()
    plt.savefig(output_dir / "average_consumption_by_hour.png", dpi=150)
    plt.close()

    print(f"Raw rows read: {len(raw):,}")
    print(f"15-minute rows prepared: {len(data):,}")
    print(f"Saved processed dataset: {processed_path.resolve()}")
    return data


CLASSICAL_FEATURES = [
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


# ===============================================
# CLASSICAL ANOMALY DETECTION — STATISTICAL BASELINE + ISOLATION FOREST
# ===============================================

def chronological_split(df: pd.DataFrame, train_fraction: float):
    split_at = int(len(df) * train_fraction)
    if split_at < 100 or len(df) - split_at < 20:
        raise ValueError("Not enough data for train/test split.")
    return df.iloc[:split_at].copy(), df.iloc[split_at:].copy()


def detect_classical_anomalies(data: pd.DataFrame, output_dir: Path,
                      train_fraction: float = 0.8,
                      contamination: float = 0.01,
                      threshold_quantile: float = 0.99):
    print("\n" + "=" * 68)
    print("CLASSICAL METHODS: STATISTICAL BASELINE + ISOLATION FOREST")
    print("=" * 68)

    df = data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    train, test = chronological_split(df, train_fraction)

    # Statistical baseline: expected median power for hour and weekend status.
    baseline_table = train.groupby(["hour", "is_weekend"])["Global_active_power"].median()
    global_median = float(train["Global_active_power"].median())

    def expected_for(frame):
        keys = pd.MultiIndex.from_frame(frame[["hour", "is_weekend"]])
        values = baseline_table.reindex(keys).to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, global_median)

    train_expected = expected_for(train)
    test_expected = expected_for(test)
    train_errors = np.abs(train["Global_active_power"].to_numpy() - train_expected)
    test_errors = np.abs(test["Global_active_power"].to_numpy() - test_expected)
    stat_threshold = float(np.quantile(train_errors, threshold_quantile))

    stat_results = test[
        ["timestamp", "Global_active_power", "hour", "day_of_week", "is_weekend"]
    ].copy()
    stat_results["expected_consumption"] = test_expected
    stat_results["absolute_deviation"] = test_errors
    stat_results["anomaly_threshold"] = stat_threshold
    stat_results["anomaly_score"] = test_errors / max(stat_threshold, 1e-8)
    stat_results["is_anomaly"] = test_errors > stat_threshold
    stat_results.to_csv(output_dir / "statistical_baseline_results.csv", index=False)

    # Isolation Forest; train-only imputation/scaling/model fitting.
    features = [c for c in CLASSICAL_FEATURES if c in df.columns]
    train_x = train[features].replace([np.inf, -np.inf], np.nan)
    test_x = test[features].replace([np.inf, -np.inf], np.nan)
    medians = train_x.median()
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = scaler.transform(test_x)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_scaled)

    train_scores = -model.decision_function(train_scaled)
    test_scores = -model.decision_function(test_scaled)
    # Use the training score quantile as the decision threshold.
    if_threshold = float(np.quantile(train_scores, 1 - contamination))

    if_results = test[["timestamp", "Global_active_power"]].copy()
    if_results["anomaly_score"] = test_scores
    if_results["anomaly_threshold"] = if_threshold
    if_results["is_anomaly"] = test_scores > if_threshold
    if_results["model"] = "IsolationForest"
    if_results.to_csv(output_dir / "isolation_forest_results.csv", index=False)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": features,
            "training_medians": medians,
            "score_threshold": if_threshold,
        },
        output_dir / "isolation_forest_model.joblib",
    )

    comparison = pd.DataFrame({
        "method": ["Statistical baseline", "Isolation Forest"],
        "test_rows": [len(test), len(test)],
        "flagged_rows": [
            int(stat_results["is_anomaly"].sum()),
            int(if_results["is_anomaly"].sum()),
        ],
    })
    comparison["flagged_percent"] = 100 * comparison["flagged_rows"] / comparison["test_rows"]
    comparison.to_csv(output_dir / "classical_model_comparison.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(comparison["method"], comparison["flagged_percent"])
    plt.ylabel("Flagged test readings (%)")
    plt.title("Classical anomaly-candidate rate by method")
    plt.tight_layout()
    plt.savefig(output_dir / "classical_model_comparison.png", dpi=150)
    plt.close()

    print(comparison.to_string(index=False))
    print("Classical anomaly detection complete. Results are candidate anomalies, not ground-truth labels.")
    return train, test


# ================================================
# DEEP LEARNING ANOMALY DETECTION — LSTM PREDICTION + LSTM AUTOENCODER
# ================================================

def make_sequences(values: np.ndarray, window: int):
    """Overlapping windows: (samples, time steps, 1 feature)."""
    if len(values) <= window:
        raise ValueError(f"Need more than {window} rows to create sequences.")
    return np.stack([values[i - window:i] for i in range(window, len(values))])


def build_next_step_predictor(window: int):
    # Given the previous window readings, predict the next reading.
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_sequence_autoencoder(window: int):
    from tensorflow.keras import layers, models
    inputs = layers.Input(shape=(window, 1))
    encoded = layers.LSTM(32, return_sequences=False)(inputs)
    repeated = layers.RepeatVector(window)(encoded)
    decoded = layers.LSTM(32, return_sequences=True)(repeated)
    outputs = layers.TimeDistributed(layers.Dense(1))(decoded)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def detect_lstm_anomalies(data: pd.DataFrame, output_dir: Path,
                          train_fraction: float = 0.8,
                          window: int = 4,
                          epochs: int = 20,
                          batch_size: int = 128,
                          threshold_quantile: float = 0.99,
                          seed: int = 42):
    print("\n" + "=" * 68)
    print("DEEP LEARNING METHODS: LSTM PREDICTION + LSTM AUTOENCODER")
    print("=" * 68)

    # TensorFlow is imported here so Members 1 and 2 can run without it.
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for the LSTM-based models. Install it using "
            "'pip install tensorflow' or run in Google Colab."
        ) from exc

    np.random.seed(seed)
    tf.random.set_seed(seed)

    df = data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    train, test = chronological_split(df, train_fraction)

    scaler = MinMaxScaler()
    train_values = scaler.fit_transform(train[["Global_active_power"]].astype(float))
    test_values = scaler.transform(test[["Global_active_power"]].astype(float))
    joblib.dump(scaler, output_dir / "power_scaler.joblib")

    early_stop = EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    # A) LSTM next-step prediction.
    x_train_pred = make_sequences(train_values, window)
    y_train_pred = train_values[window:, 0]
    predictor = build_next_step_predictor(window)
    predictor.fit(
        x_train_pred, y_train_pred,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1,
    )
    predictor.save(output_dir / "lstm_prediction_model.keras")

    # Include the final training window to predict the first test reading.
    combined = np.vstack([train_values[-window:], test_values])
    x_test_pred = make_sequences(combined, window)
    pred_train = predictor.predict(x_train_pred, verbose=0).reshape(-1)
    pred_test = predictor.predict(x_test_pred, verbose=0).reshape(-1)

    train_pred_error = np.abs(y_train_pred - pred_train)
    pred_threshold = float(np.quantile(train_pred_error, threshold_quantile))
    test_pred_error = np.abs(test_values[:, 0] - pred_test)

    actual_kw = scaler.inverse_transform(test_values).reshape(-1)
    predicted_kw = scaler.inverse_transform(pred_test.reshape(-1, 1)).reshape(-1)
    pred_results = pd.DataFrame({
        "timestamp": test["timestamp"].to_numpy(),
        "actual_consumption_kw": actual_kw,
        "predicted_consumption_kw": predicted_kw,
        "prediction_error_scaled": test_pred_error,
        "anomaly_threshold_scaled": pred_threshold,
        "anomaly_score": test_pred_error / max(pred_threshold, 1e-8),
        "is_anomaly": test_pred_error > pred_threshold,
        "model": "LSTM_prediction",
    })
    pred_results.to_csv(output_dir / "lstm_prediction_results.csv", index=False)

    # Regression evaluation in the original kW scale (not anomaly-classification accuracy).
    prediction_mae = float(mean_absolute_error(
        pred_results["actual_consumption_kw"], pred_results["predicted_consumption_kw"]
    ))
    prediction_rmse = float(np.sqrt(mean_squared_error(
        pred_results["actual_consumption_kw"], pred_results["predicted_consumption_kw"]
    )))

    # Prediction-error visualization. Larger errors may be flagged as candidates.
    prediction_residuals = (
        pred_results["actual_consumption_kw"] - pred_results["predicted_consumption_kw"]
    )
    plt.figure(figsize=(10, 5))
    plt.hist(prediction_residuals, bins=50)
    plt.title("LSTM prediction residual distribution (test set)")
    plt.xlabel("Actual - predicted active power (kW)")
    plt.ylabel("Number of readings")
    plt.tight_layout()
    plt.savefig(output_dir / "lstm_prediction_error_distribution.png", dpi=150)
    plt.close()

    # B) LSTM autoencoder. Each input sequence is trained to reconstruct itself.
    x_train_ae = make_sequences(train_values, window)
    x_test_ae = make_sequences(combined, window)
    autoencoder = build_sequence_autoencoder(window)
    autoencoder.fit(
        x_train_ae, x_train_ae,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1,
    )
    autoencoder.save(output_dir / "lstm_autoencoder_model.keras")

    train_reconstructed = autoencoder.predict(x_train_ae, verbose=0)
    test_reconstructed = autoencoder.predict(x_test_ae, verbose=0)
    train_ae_error = np.mean(np.square(x_train_ae - train_reconstructed), axis=(1, 2))
    test_ae_error = np.mean(np.square(x_test_ae - test_reconstructed), axis=(1, 2))
    ae_threshold = float(np.quantile(train_ae_error, threshold_quantile))

    ae_results = pd.DataFrame({
        "timestamp": test["timestamp"].to_numpy(),
        "actual_consumption_kw": test["Global_active_power"].to_numpy(dtype=float),
        "reconstruction_error": test_ae_error,
        "anomaly_threshold": ae_threshold,
        "anomaly_score": test_ae_error / max(ae_threshold, 1e-12),
        "is_anomaly": test_ae_error > ae_threshold,
        "model": "LSTM_autoencoder",
    })
    ae_results.to_csv(output_dir / "lstm_autoencoder_results.csv", index=False)

    # Reconstruction MAE/RMSE for the ending reading of each test window, in kW.
    # These are reconstruction-error metrics, not anomaly-classification accuracy.
    ae_actual_kw = scaler.inverse_transform(
        x_test_ae[:, -1, 0].reshape(-1, 1)
    ).reshape(-1)
    ae_reconstructed_kw = scaler.inverse_transform(
        test_reconstructed[:, -1, 0].reshape(-1, 1)
    ).reshape(-1)
    ae_reconstruction_mae = float(mean_absolute_error(ae_actual_kw, ae_reconstructed_kw))
    ae_reconstruction_rmse = float(np.sqrt(mean_squared_error(ae_actual_kw, ae_reconstructed_kw)))

    evaluation_metrics = pd.DataFrame([
        {
            "model": "LSTM prediction",
            "metric_scope": "Next-step prediction on test set",
            "MAE_kW": prediction_mae,
            "RMSE_kW": prediction_rmse,
        },
        {
            "model": "LSTM autoencoder",
            "metric_scope": "Reconstruction of ending reading in each test window",
            "MAE_kW": ae_reconstruction_mae,
            "RMSE_kW": ae_reconstruction_rmse,
        },
    ])
    evaluation_metrics.to_csv(output_dir / "model_evaluation_metrics.csv", index=False)

    comparison = pd.DataFrame({
        "model": ["LSTM prediction", "LSTM autoencoder"],
        "test_rows": [len(pred_results), len(ae_results)],
        "flagged_rows": [
            int(pred_results["is_anomaly"].sum()),
            int(ae_results["is_anomaly"].sum()),
        ],
    })
    comparison["flagged_percent"] = 100 * comparison["flagged_rows"] / comparison["test_rows"]
    comparison.to_csv(output_dir / "deep_learning_model_comparison.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(comparison["model"], comparison["flagged_percent"])
    plt.ylabel("Flagged test readings (%)")
    plt.title("LSTM anomaly-candidate rate")
    plt.tight_layout()
    plt.savefig(output_dir / "deep_learning_model_comparison.png", dpi=150)
    plt.close()

    # Actual vs predicted for up to one test week.
    n = min(7 * 96, len(pred_results))
    plt.figure(figsize=(14, 5))
    plt.plot(
        pred_results["timestamp"].iloc[:n],
        pred_results["actual_consumption_kw"].iloc[:n],
        label="Actual", linewidth=1,
    )
    plt.plot(
        pred_results["timestamp"].iloc[:n],
        pred_results["predicted_consumption_kw"].iloc[:n],
        label="LSTM predicted", linewidth=1,
    )
    flagged = pred_results.iloc[:n]
    flagged = flagged[flagged["is_anomaly"]]
    if not flagged.empty:
        plt.scatter(
            flagged["timestamp"], flagged["actual_consumption_kw"],
            label="Flagged candidate", s=18,
        )
    plt.title("LSTM prediction: actual vs predicted (first test week)")
    plt.xlabel("Timestamp")
    plt.ylabel("Active power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lstm_prediction_plot.png", dpi=150)
    plt.close()

    metadata = {
        "window_15min_steps": window,
        "window_minutes": window * 15,
        "train_fraction": train_fraction,
        "threshold_quantile": threshold_quantile,
        "prediction_threshold_scaled": pred_threshold,
        "autoencoder_threshold": ae_threshold,
        "note": "Flags are anomaly candidates; dataset has no complete ground-truth anomaly labels.",
    }
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(comparison.to_string(index=False))
    print("\nTest-set regression/reconstruction metrics (kW):")
    print(evaluation_metrics.to_string(index=False))
    print(f"Saved evaluation metrics: {(output_dir / 'model_evaluation_metrics.csv').resolve()}")
    print("Note: MAE/RMSE measure prediction/reconstruction error, not anomaly-classification accuracy.")
    print("LSTM anomaly detection complete. Saved models, result CSVs and graphs.")
    print("Inspect candidate events before interpreting them as genuine anomalies.")


# =================
# RUN ANOMALY DETECTION
# =================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare raw household electricity data and run classical + LSTM anomaly detection."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the original UCI household_power_consumption.txt file",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--window", type=int, default=4,
                        help="Sequence window; 4 steps = 1 hour at 15-minute resolution")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start from the raw UCI dataset: preprocessing is included in this file.
    processed = prepare_electricity_data(args.input, output_dir)

    detect_classical_anomalies(
        processed, output_dir,
        train_fraction=args.train_fraction,
        contamination=args.contamination,
        threshold_quantile=args.threshold_quantile,
    )
    detect_lstm_anomalies(
        processed, output_dir,
        train_fraction=args.train_fraction,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threshold_quantile=args.threshold_quantile,
    )

    print("\n" + "=" * 68)
    print("COMPLETE PIPELINE FINISHED")
    print(f"Processed data, graphs, models and results saved in: {output_dir.resolve()}")
    print("Flags are candidate anomalies, not verified fault labels.")
    print("=" * 68)


if __name__ == "__main__":
    main()
