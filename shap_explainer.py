"""
SHAP Explainability Module for Household Electricity Anomaly Detection
Part of Member 4: Context-Aware Explainable AI (XAI) Architecture.

This module integrates TreeSHAP with the serialized Isolation Forest model,
formulating and calculating local and global game-theoretic feature attributions.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class IsolationForestSHAPExplainer:
    """
    Wrapper for computing and interpreting Shapley Additive Explanations (SHAP)
    on the trained Isolation Forest anomaly detector.
    """

    def __init__(self, model_joblib_path: Path):
        self.model_data = joblib.load(model_joblib_path)
        self.model = self.model_data["model"]
        self.scaler = self.model_data["scaler"]
        self.feature_columns = self.model_data["feature_columns"]
        self.training_medians = self.model_data["training_medians"]
        self.score_threshold = self.model_data["score_threshold"]
        
        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        self.expected_path_length = float(self.explainer.expected_value[0])

    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Impute missing feature values using training medians and standardize.
        """
        x = df[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        x = x.fillna(self.training_medians)
        return self.scaler.transform(x)

    def explain_instances(
        self, 
        df_instances: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Shapley values for input observations.
        
        Returns:
          - shap_path_lengths: Shapley values explaining tree path length h(x).
          - shap_anomaly_attributions: Negated values explaining anomaly severity (-h(x)).
          - scaled_feature_matrix: Standardized features input to the trees.
        """
        scaled_x = self.preprocess_features(df_instances)
        shap_path_lengths = self.explainer.shap_values(scaled_x)
        
        # Invert path length: negative contribution to path length means
        # shorter path (faster isolation), which contributes POSITIVELY to anomaly severity.
        shap_anomaly_attributions = -shap_path_lengths
        return shap_path_lengths, shap_anomaly_attributions, scaled_x

    def get_feature_contributions_summary(
        self,
        df_instances: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract mean absolute SHAP contributions across instances.
        """
        _, shap_anom, _ = self.explain_instances(df_instances)
        mean_abs_impact = np.mean(np.abs(shap_anom), axis=0)
        
        ranking_df = pd.DataFrame({
            "feature": self.feature_columns,
            "mean_abs_shap_attribution": mean_abs_impact,
        }).sort_values("mean_abs_shap_attribution", ascending=False).reset_index(drop=True)
        
        return ranking_df

    def plot_summary(
        self,
        df_instances: pd.DataFrame,
        output_path: Path,
        max_display: int = 12
    ):
        """
        Generate and save a publication-ready SHAP summary plot.
        """
        _, shap_anom, scaled_x = self.explain_instances(df_instances)
        
        plt.figure(figsize=(10, 6))
        # Create a summary bar plot of mean absolute anomaly attribution
        mean_impact = np.mean(np.abs(shap_anom), axis=0)
        indices = np.argsort(mean_impact)[-max_display:]
        
        plt.barh(
            [self.feature_columns[i] for i in indices],
            mean_impact[indices],
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.8
        )
        plt.xlabel("Mean |SHAP Value| (Impact on Anomaly Severity)", fontsize=11)
        plt.title("TreeSHAP Global Feature Importance for Candidate Anomalies", fontsize=12, fontweight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    def plot_individual_explanation(
        self,
        df_single_row: pd.DataFrame,
        output_path: Path,
        top_k: int = 8
    ):
        """
        Generate a horizontal waterfall-style attribution plot for an individual anomaly.
        """
        timestamp = str(df_single_row["timestamp"].iloc[0])
        actual_power = float(df_single_row["Global_active_power"].iloc[0])
        
        _, shap_anom, scaled_x = self.explain_instances(df_single_row)
        attributions = shap_anom[0]
        
        # Sort by absolute impact
        sorted_idx = np.argsort(np.abs(attributions))[-top_k:]
        top_features = [self.feature_columns[i] for i in sorted_idx]
        top_values = attributions[sorted_idx]
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_values]
        
        plt.figure(figsize=(9, 5))
        bars = plt.barh(top_features, top_values, color=colors, edgecolor="black", linewidth=0.8)
        plt.axvline(0, color="black", linewidth=1.0, linestyle="-")
        
        # Label bars
        for bar, val in zip(bars, top_values):
            sign = "+" if val > 0 else ""
            offset = 0.05 if val >= 0 else -0.15
            plt.text(
                val + offset, 
                bar.get_y() + bar.get_height() / 2, 
                f"{sign}{val:.2f}", 
                va="center", 
                fontsize=9, 
                fontweight="bold"
            )
            
        plt.xlabel("SHAP Attribution to Anomaly Severity (Positive = Increases Outlier Score)", fontsize=10)
        plt.title(
            f"Local TreeSHAP Attribution for Flagged Event at {timestamp}\nActual Power: {actual_power:.2f} kW", 
            fontsize=11, 
            fontweight="bold"
        )
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

