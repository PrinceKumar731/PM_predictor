from __future__ import annotations

import joblib
import argparse

from .config import MODELS_DIR, TARGET_COLUMN, ensure_directories
from .forecast_model import (
    build_forecast_model,
    build_forecast_training_frame,
    future_safe_feature_columns,
    load_feature_frame,
    save_forecast_artifacts,
)
from .modeling import chronological_split, evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dedicated future forecast model.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in months.")
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()
    frame = load_feature_frame()
    train_frame = build_forecast_training_frame(frame, horizon=args.horizon)
    features = future_safe_feature_columns(train_frame)
    split_frame = train_frame.drop(columns=["time"]).rename(columns={"target_time": "time"})
    splits = chronological_split(split_frame)

    # Re-slice from original train_frame to preserve target_next_pm25.
    train_times = set(splits.train["time"])
    val_times = set(splits.validation["time"])
    test_times = set(splits.test["time"])
    train_part = train_frame[train_frame["target_time"].isin(train_times)].copy()
    val_part = train_frame[train_frame["target_time"].isin(val_times)].copy()
    test_part = train_frame[train_frame["target_time"].isin(test_times)].copy()

    model = build_forecast_model()
    model.fit(
        train_part[features],
        train_part["target_next_pm25"],
        eval_set=[(train_part[features], train_part["target_next_pm25"]), (val_part[features], val_part["target_next_pm25"])],
        verbose=False,
    )

    val_pred = model.predict(val_part[features])
    test_pred = model.predict(test_part[features])
    val_metrics = evaluate_predictions(val_part["target_next_pm25"], val_pred)
    test_metrics = evaluate_predictions(test_part["target_next_pm25"], test_pred)

    save_forecast_artifacts(model, features, horizon=args.horizon)
    joblib.dump({"validation": val_metrics, "test": test_metrics}, MODELS_DIR / f"forecast_h{args.horizon}_metrics.joblib")

    print(f"Forecast horizon: {args.horizon} month(s)")
    print("Forecast validation metrics:", val_metrics)
    print(f"Forecast validation accuracy: {val_metrics['accuracy_pct']:.2f}%")
    print("Forecast test metrics:", test_metrics)
    print(f"Forecast test accuracy: {test_metrics['accuracy_pct']:.2f}%")
    print(f"Saved forecast model to {MODELS_DIR / f'pm25_forecast_h{args.horizon}_xgboost.joblib'}")


if __name__ == "__main__":
    main()
