from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_predicted_vs_actual(results: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 7))
    sns.scatterplot(data=results, x="actual", y="predicted", alpha=0.7)
    min_val = min(results["actual"].min(), results["predicted"].min())
    max_val = max(results["actual"].max(), results["predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("Predicted vs Actual PM2.5")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residual_distribution(results: pd.DataFrame, output_path: Path) -> None:
    residuals = results["actual"] - results["predicted"]
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_heatmap(frame: pd.DataFrame, value_column: str, title: str, output_path: Path) -> None:
    pivot = frame.pivot_table(index="latitude", columns="longitude", values=value_column, aggfunc="mean")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot.sort_index(ascending=False), cmap="YlOrRd", linewidths=0.2)
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def pm25_to_aqi_category(values: pd.Series | np.ndarray) -> pd.Categorical:
    bins = [-np.inf, 12.0, 35.4, 55.4, 150.4, 250.4, np.inf]
    labels = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True)


def plot_training_curves(history: dict, output_path: Path) -> None:
    train_rmse = history.get("validation_0", {}).get("rmse", [])
    val_rmse = history.get("validation_1", {}).get("rmse", [])
    train_mae = history.get("validation_0", {}).get("mae", train_rmse)
    val_mae = history.get("validation_1", {}).get("mae", val_rmse)
    epochs = np.arange(1, len(train_rmse) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_rmse, label="Train RMSE", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, val_rmse, label="Validation RMSE", color="#ff7f0e", linestyle="--", linewidth=2)
    axes[0].set_title("RMSE Over Training")
    axes[0].set_xlabel("Boosting Round")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()

    axes[1].plot(epochs, train_mae, label="Train MAE", color="#1f77b4", linewidth=2)
    axes[1].plot(epochs, val_mae, label="Validation MAE", color="#ff7f0e", linestyle="--", linewidth=2)
    axes[1].set_title("MAE Over Training")
    axes[1].set_xlabel("Boosting Round")
    axes[1].set_ylabel("MAE")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_aqi_confusion_matrix(actual: pd.Series, predicted: pd.Series, output_path: Path) -> None:
    actual_cat = pm25_to_aqi_category(actual)
    pred_cat = pm25_to_aqi_category(predicted)
    labels = list(actual_cat.cat.categories)
    matrix = confusion_matrix(actual_cat, pred_cat, labels=labels, normalize="true")

    plt.figure(figsize=(8.5, 6.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.xlabel("Predicted AQI Category")
    plt.ylabel("Actual AQI Category")
    plt.title("AQI Category Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_results_board(metrics: dict[str, float], history: dict, actual: pd.Series, predicted: pd.Series, output_path: Path) -> None:
    actual_cat = pm25_to_aqi_category(actual)
    pred_cat = pm25_to_aqi_category(predicted)
    labels = list(actual_cat.cat.categories)
    matrix = confusion_matrix(actual_cat, pred_cat, labels=labels, normalize="true")

    train_rmse = history.get("validation_0", {}).get("rmse", [])
    val_rmse = history.get("validation_1", {}).get("rmse", [])
    train_mae = history.get("validation_0", {}).get("mae", train_rmse)
    val_mae = history.get("validation_1", {}).get("mae", val_rmse)
    epochs = np.arange(1, len(train_rmse) + 1)

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[0.18, 0.82], width_ratios=[1, 1, 1.45])

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.5, 0.62, "Results", ha="center", va="center", fontsize=28, fontweight="bold")
    title_ax.axhline(0.15, color="#444444", linewidth=1.2, xmin=0.02, xmax=0.98)

    ax_rmse = fig.add_subplot(grid[1, 0])
    ax_rmse.plot(epochs, train_rmse, color="#2C6FB7", linewidth=2.5, marker="o", markersize=3)
    ax_rmse.plot(epochs, val_rmse, color="#F28E2B", linewidth=2, linestyle="--")
    ax_rmse.set_title("RMSE Over Training", fontsize=16)
    ax_rmse.set_xlabel("Boosting Round")
    ax_rmse.set_ylabel("RMSE")

    ax_mae = fig.add_subplot(grid[1, 1])
    ax_mae.plot(epochs, train_mae, color="#2C6FB7", linewidth=2.5, marker="o", markersize=3)
    ax_mae.plot(epochs, val_mae, color="#F28E2B", linewidth=2, linestyle="--")
    ax_mae.set_title("MAE Over Training", fontsize=16)
    ax_mae.set_xlabel("Boosting Round")
    ax_mae.set_ylabel("MAE")

    ax_cm = fig.add_subplot(grid[1, 2])
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax_cm,
    )
    ax_cm.set_title("AQI Confusion Matrix", fontsize=16)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    fig.text(0.20, 0.08, "Training Error Curves", ha="center", fontsize=19, fontweight="bold")
    fig.text(0.73, 0.08, "AQI Category Agreement", ha="center", fontsize=19, fontweight="bold")
    fig.text(
        0.5,
        0.92,
        f"Test RMSE: {metrics['rmse']:.2f}   |   Test MAE: {metrics['mae']:.2f}   |   Test R²: {metrics['r2']:.3f}   |   Accuracy: {metrics['accuracy_pct']:.2f}%",
        ha="center",
        fontsize=13,
        color="#333333",
    )

    fig.tight_layout(rect=[0, 0.1, 1, 0.92])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
