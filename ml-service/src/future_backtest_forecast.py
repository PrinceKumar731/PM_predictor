from __future__ import annotations

import argparse

import pandas as pd

from .config import FEATURE_TABLE_NAME, PROCESSED_DIR, TARGET_COLUMN, ensure_directories
from .forecast_model import (
    build_forecast_model,
    build_forecast_training_frame,
    future_safe_feature_columns,
)
from .predict_future import blend_future_prediction, build_future_month_rows, next_month


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest dedicated one-step forecast model on unseen future years.")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    frame = pd.read_parquet(PROCESSED_DIR / FEATURE_TABLE_NAME).sort_values("time").reset_index(drop=True)
    args = parse_args()

    results: list[dict[str, float | int]] = []

    for test_year in range(args.start_year, args.end_year + 1):
        train_frame = frame[frame["year"] < test_year].copy()
        actual_year_frame = frame[frame["year"] == test_year].copy()
        if train_frame.empty or actual_year_frame.empty:
            continue

        forecast_train = build_forecast_training_frame(train_frame)
        features = future_safe_feature_columns(forecast_train)
        model = build_forecast_model()
        model.fit(forecast_train[features], forecast_train["target_next_pm25"], verbose=False)

        history_columns = ["time", "latitude", "longitude", "pm25"]
        if "satellite_pm25_aux" in train_frame.columns:
            history_columns.append("satellite_pm25_aux")
        history = train_frame[history_columns].copy()
        observed_history = history.copy()

        for month in range(1, 13):
            target_time = pd.Timestamp(year=test_year, month=month, day=1)
            forecast_time = next_month(history["time"].max())
            horizon_steps = 1
            while forecast_time <= target_time:
                future_rows = build_future_month_rows(history, forecast_time, features)
                raw_predictions = model.predict(future_rows[features])
                blended_predictions = []
                for idx, prediction in enumerate(raw_predictions):
                    row = future_rows.iloc[idx]
                    location_history = observed_history[
                        (observed_history["latitude"] == row["latitude"]) & (observed_history["longitude"] == row["longitude"])
                    ].copy()
                    blended_predictions.append(
                        blend_future_prediction(
                            model_prediction=float(prediction),
                            location_history=location_history.sort_values("time"),
                            forecast_time=forecast_time,
                            horizon_steps=horizon_steps,
                        )
                    )
                future_rows["pm25"] = blended_predictions
                append_columns = ["time", "latitude", "longitude", "pm25"]
                if "satellite_pm25_aux" in future_rows.columns:
                    append_columns.append("satellite_pm25_aux")
                history = pd.concat([history, future_rows[append_columns]], ignore_index=True)
                forecast_time = next_month(forecast_time)
                horizon_steps += 1

            pred_month = history[history["time"] == target_time].copy()
            actual_month = actual_year_frame[actual_year_frame["month"] == month].copy()
            if pred_month.empty or actual_month.empty:
                continue

            pred_month["distance_to_query"] = ((pred_month["latitude"] - args.lat) ** 2 + (pred_month["longitude"] - args.lon) ** 2) ** 0.5
            actual_month["distance_to_query"] = ((actual_month["latitude"] - args.lat) ** 2 + (actual_month["longitude"] - args.lon) ** 2) ** 0.5
            pred_row = pred_month.sort_values("distance_to_query").iloc[0]
            actual_row = actual_month.sort_values("distance_to_query").iloc[0]

            predicted = float(pred_row["pm25"])
            actual = float(actual_row[TARGET_COLUMN])
            absolute_error = abs(predicted - actual)
            accuracy = max(0.0, 100.0 - (absolute_error / actual * 100.0)) if actual != 0 else 0.0
            results.append(
                {
                    "year": test_year,
                    "month": month,
                    "predicted_pm25": predicted,
                    "actual_pm25": actual,
                    "absolute_error": absolute_error,
                    "accuracy_pct": accuracy,
                }
            )

    results_frame = pd.DataFrame(results)
    output_path = PROCESSED_DIR / "future_backtest_forecast_results.csv"
    results_frame.to_csv(output_path, index=False)
    for year, year_frame in results_frame.groupby("year"):
        print(
            f"{int(year)}: Mean Abs Error={year_frame['absolute_error'].mean():.2f}, "
            f"Average Accuracy={year_frame['accuracy_pct'].mean():.2f}%"
        )
    if not results_frame.empty:
        print(
            f"OVERALL: Mean Abs Error={results_frame['absolute_error'].mean():.2f}, "
            f"Average Accuracy={results_frame['accuracy_pct'].mean():.2f}%"
        )
        print(f"Saved dedicated forecast backtest results to {output_path}")


if __name__ == "__main__":
    main()
