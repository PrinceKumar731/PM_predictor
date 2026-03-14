from __future__ import annotations

import argparse

import pandas as pd

from .config import FEATURE_TABLE_NAME, PROCESSED_DIR, TARGET_COLUMN, ensure_directories
from .forecast_model import build_forecast_model, build_forecast_training_frame, future_safe_feature_columns
from .predict_future import blend_future_prediction, build_future_month_rows, next_month
from .predict_future_forecast import select_horizon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate future forecast bias using historical unseen-year backtests.")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2023)
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()
    frame = pd.read_parquet(PROCESSED_DIR / FEATURE_TABLE_NAME).sort_values("time").reset_index(drop=True)

    residual_rows: list[dict[str, float | int]] = []

    for test_year in range(args.start_year, args.end_year + 1):
        train_frame = frame[frame["year"] < test_year].copy()
        actual_year_frame = frame[frame["year"] == test_year].copy()
        if train_frame.empty or actual_year_frame.empty:
            continue

        models_by_horizon: dict[int, tuple[object, list[str]]] = {}
        for horizon in [1, 3, 6]:
            forecast_train = build_forecast_training_frame(train_frame, horizon=horizon)
            features = future_safe_feature_columns(forecast_train)
            model = build_forecast_model()
            model.fit(forecast_train[features], forecast_train["target_next_pm25"], verbose=False)
            models_by_horizon[horizon] = (model, features)

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
                selected_horizon = select_horizon(horizon_steps)
                model, features = models_by_horizon[selected_horizon]
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

            pred_month = history[history["time"] == target_time][["latitude", "longitude", "pm25"]].copy()
            actual_month = actual_year_frame[actual_year_frame["month"] == month][["latitude", "longitude", TARGET_COLUMN]].copy()
            if pred_month.empty or actual_month.empty:
                continue

            pred_month = pred_month.rename(columns={"pm25": "predicted_pm25"})
            actual_month = actual_month.rename(columns={TARGET_COLUMN: "actual_pm25"})
            pred_month["latitude"] = pred_month["latitude"].round(4)
            pred_month["longitude"] = pred_month["longitude"].round(4)
            actual_month["latitude"] = actual_month["latitude"].round(4)
            actual_month["longitude"] = actual_month["longitude"].round(4)
            merged = pred_month.merge(actual_month, on=["latitude", "longitude"], how="inner")
            if merged.empty:
                continue

            merged["residual"] = merged["predicted_pm25"] - merged["actual_pm25"]
            residual_rows.extend(
                [
                    {"year": test_year, "month": month, "residual": float(value)}
                    for value in merged["residual"].tolist()
                ]
            )

    residual_frame = pd.DataFrame(residual_rows)
    monthly_bias = (
        residual_frame.groupby("month", as_index=False)
        .agg(monthly_bias=("residual", "mean"), monthly_abs_error=("residual", lambda s: float(s.abs().mean())))
        .sort_values("month")
        .reset_index(drop=True)
    )

    output_path = PROCESSED_DIR / "forecast_monthly_bias.csv"
    monthly_bias.to_csv(output_path, index=False)

    print(monthly_bias.to_string(index=False))
    print(f"Saved monthly forecast bias calibration to {output_path}")


if __name__ == "__main__":
    main()
