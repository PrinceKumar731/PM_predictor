from __future__ import annotations

import json

import joblib
import pandas as pd

from .config import FEATURE_TABLE_NAME, MODELS_DIR, PLOTS_DIR, PROCESSED_DIR, TARGET_COLUMN, ensure_directories
from .modeling import chronological_split, evaluate_predictions
from .visualization import plot_aqi_confusion_matrix, plot_results_board, plot_training_curves


def main() -> None:
    ensure_directories()
    frame = pd.read_parquet(PROCESSED_DIR / FEATURE_TABLE_NAME).sort_values("time").reset_index(drop=True)
    model = joblib.load(MODELS_DIR / "pm25_xgboost.joblib")
    features = joblib.load(MODELS_DIR / "feature_columns.joblib")
    with open(MODELS_DIR / "training_history.json", "r", encoding="utf-8") as file:
        history = json.load(file)

    splits = chronological_split(frame)
    X_test = splits.test[features]
    y_test = splits.test[TARGET_COLUMN]
    pred_test = model.predict(X_test)
    metrics = evaluate_predictions(y_test, pred_test)

    plot_training_curves(history, PLOTS_DIR / "training_curves.png")
    plot_aqi_confusion_matrix(y_test, pd.Series(pred_test), PLOTS_DIR / "aqi_confusion_matrix.png")
    plot_results_board(
        metrics=metrics,
        history=history,
        actual=y_test.reset_index(drop=True),
        predicted=pd.Series(pred_test),
        output_path=PLOTS_DIR / "results_board.png",
    )

    print(f"Saved training curves to {PLOTS_DIR / 'training_curves.png'}")
    print(f"Saved AQI confusion matrix to {PLOTS_DIR / 'aqi_confusion_matrix.png'}")
    print(f"Saved composite results board to {PLOTS_DIR / 'results_board.png'}")


if __name__ == "__main__":
    main()
