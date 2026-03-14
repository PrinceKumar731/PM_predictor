from __future__ import annotations

import argparse

import pandas as pd

from .config import FEATURE_TABLE_NAME, PROCESSED_DIR
from .forecast_model import load_forecast_artifacts
from .predict_future import blend_future_prediction, build_future_month_rows, next_month


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast future PM2.5 using the dedicated one-step forecast model.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--actual", type=float, required=False)
    return parser.parse_args()


def select_horizon(horizon_steps: int) -> int:
    if horizon_steps <= 1:
        return 1
    if horizon_steps <= 3:
        return 3
    return 6


def load_monthly_bias_map() -> dict[int, float]:
    path = PROCESSED_DIR / "forecast_monthly_bias.csv"
    if not path.exists():
        return {}
    bias_frame = pd.read_csv(path)
    return {int(row["month"]): float(row["monthly_bias"]) for _, row in bias_frame.iterrows()}


def main() -> None:
    args = parse_args()
    target_time = pd.Timestamp(year=args.year, month=args.month, day=1)

    history_frame = pd.read_parquet(PROCESSED_DIR / FEATURE_TABLE_NAME)
    history_columns = ["time", "latitude", "longitude", "pm25"]
    if "satellite_pm25_aux" in history_frame.columns:
        history_columns.append("satellite_pm25_aux")
    history = history_frame[history_columns].copy()
    history["time"] = pd.to_datetime(history["time"])
    history = history.sort_values(["time", "latitude", "longitude"]).reset_index(drop=True)
    observed_history = history.copy()
    monthly_bias = load_monthly_bias_map()

    last_available = history["time"].max()
    if target_time <= last_available:
        raise ValueError("Requested month is inside the dataset. Use src.predict instead.")

    forecast_time = next_month(last_available)
    horizon_steps = 1

    while forecast_time <= target_time:
        selected_horizon = select_horizon(horizon_steps)
        model, features = load_forecast_artifacts(selected_horizon)
        future_rows = build_future_month_rows(history, forecast_time, features)
        raw_predictions = model.predict(future_rows[features])
        future_rows["pm25"] = [
            (
                blend_future_prediction(
                    model_prediction=float(pred),
                    location_history=observed_history[
                        (observed_history["latitude"] == row["latitude"]) & (observed_history["longitude"] == row["longitude"])
                    ].sort_values("time"),
                    forecast_time=forecast_time,
                    horizon_steps=horizon_steps,
                )
                - monthly_bias.get(forecast_time.month, 0.0)
            )
            for pred, (_, row) in zip(raw_predictions, future_rows.iterrows())
        ]

        future_rows["pm25"] = [max(0.0, float(value)) for value in future_rows["pm25"]]

        append_columns = ["time", "latitude", "longitude", "pm25"]
        if "satellite_pm25_aux" in future_rows.columns:
            append_columns.append("satellite_pm25_aux")
        history = pd.concat([history, future_rows[append_columns]], ignore_index=True)
        forecast_time = next_month(forecast_time)
        horizon_steps += 1

    target_rows = history[history["time"] == target_time].copy()
    target_rows["distance_to_query"] = ((target_rows["latitude"] - args.lat) ** 2 + (target_rows["longitude"] - args.lon) ** 2) ** 0.5
    row = target_rows.sort_values("distance_to_query").iloc[0]

    print("Dedicated forecast summary")
    print(f"Forecast month: {target_time.strftime('%Y-%m')}")
    print(f"Requested location: lat={args.lat:.4f}, lon={args.lon:.4f}")
    print(f"Nearest grid cell: lat={row['latitude']:.4f}, lon={row['longitude']:.4f}")
    print(f"Forecast PM2.5: {float(row['pm25']):.3f}")
    if target_time.month in monthly_bias:
        print(f"Applied monthly bias correction: {-monthly_bias[target_time.month]:.3f}")
    if args.actual is not None:
        actual = float(args.actual)
        predicted = float(row["pm25"])
        absolute_error = abs(predicted - actual)
        accuracy = max(0.0, 100.0 - (absolute_error / actual * 100.0)) if actual != 0 else 0.0
        print(f"Actual PM2.5: {actual:.3f}")
        print(f"Absolute Error: {absolute_error:.3f}")
        print(f"Approximate Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
