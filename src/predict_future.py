from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from .config import FEATURE_TABLE_NAME, MODELS_DIR, PROCESSED_DIR
from .features import month_to_season
from .data import load_meteorological_monthly_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast future PM2.5 for Pune using recursive monthly prediction.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--actual", type=float, required=False, help="Optional true PM2.5 value for error/accuracy reporting.")
    return parser.parse_args()


def next_month(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp + pd.offsets.MonthBegin(1)


def add_month_features(frame: pd.DataFrame, first_time: pd.Timestamp) -> pd.DataFrame:
    data = frame.copy()
    data["year"] = data["time"].dt.year
    data["month"] = data["time"].dt.month
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)
    data["months_since_start"] = (data["time"].dt.year - first_time.year) * 12 + (data["time"].dt.month - first_time.month)
    return data


def compute_spatial_lag_features(
    base_rows: pd.DataFrame,
    previous_month_rows: pd.DataFrame,
    radius_degrees: float = 0.12,
    eps: float = 1e-6,
) -> pd.DataFrame:
    prev = previous_month_rows[["latitude", "longitude", "pm25"]].copy()
    coords = prev[["latitude", "longitude"]].to_numpy(dtype=float)
    values = prev["pm25"].to_numpy(dtype=float)

    spatial_mean = np.full(len(base_rows), np.nan, dtype=float)
    spatial_weighted = np.full(len(base_rows), np.nan, dtype=float)
    grid_density = np.zeros(len(base_rows), dtype=float)

    for idx, (_, row) in enumerate(base_rows.iterrows()):
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        distances = np.sqrt((coords[:, 0] - lat) ** 2 + (coords[:, 1] - lon) ** 2)
        mask = (distances > 0) & (distances <= radius_degrees) & np.isfinite(values)

        if not np.any(mask):
            continue

        neighbors = values[mask]
        neighbor_distances = distances[mask]
        weights = 1.0 / (neighbor_distances + eps)
        spatial_mean[idx] = float(np.mean(neighbors))
        spatial_weighted[idx] = float(np.average(neighbors, weights=weights))
        grid_density[idx] = int(mask.sum())

    enriched = base_rows.copy()
    enriched["spatial_lag_mean_1"] = spatial_mean
    enriched["spatial_lag_weighted_1"] = spatial_weighted
    enriched["grid_density"] = grid_density
    return enriched


def build_future_month_rows(history: pd.DataFrame, forecast_time: pd.Timestamp, feature_list: list[str]) -> pd.DataFrame:
    grid = history[["latitude", "longitude"]].drop_duplicates().sort_values(["latitude", "longitude"]).reset_index(drop=True)
    grid["time"] = forecast_time

    first_time = history["time"].min()
    future_rows = add_month_features(grid, first_time)
    future_rows["season"] = future_rows["month"].map(month_to_season)

    lag_values: dict[tuple[float, float], list[float]] = {}
    for (lat, lon), group in history.sort_values("time").groupby(["latitude", "longitude"]):
        lag_values[(float(lat), float(lon))] = group["pm25"].tail(6).tolist()

    future_rows["pm_lag_1"] = future_rows.apply(lambda row: lag_values[(float(row["latitude"]), float(row["longitude"]))][-1], axis=1)
    future_rows["pm_lag_2"] = future_rows.apply(
        lambda row: lag_values[(float(row["latitude"]), float(row["longitude"]))][-2]
        if len(lag_values[(float(row["latitude"]), float(row["longitude"]))]) >= 2
        else lag_values[(float(row["latitude"]), float(row["longitude"]))][-1],
        axis=1,
    )
    future_rows["pm_lag_3"] = future_rows.apply(
        lambda row: lag_values[(float(row["latitude"]), float(row["longitude"]))][-3]
        if len(lag_values[(float(row["latitude"]), float(row["longitude"]))]) >= 3
        else lag_values[(float(row["latitude"]), float(row["longitude"]))][-1],
        axis=1,
    )
    future_rows["rolling_pm_3"] = future_rows.apply(
        lambda row: float(np.mean(lag_values[(float(row["latitude"]), float(row["longitude"]))][-3:])),
        axis=1,
    )
    future_rows["rolling_pm_6"] = future_rows.apply(
        lambda row: float(np.mean(lag_values[(float(row["latitude"]), float(row["longitude"]))][-6:])),
        axis=1,
    )
    future_rows["trend_1_3"] = (future_rows["pm_lag_1"] - future_rows["pm_lag_3"]) / 2.0

    if "satellite_pm25_aux" in history.columns:
        satellite_lag_values: dict[tuple[float, float], list[float]] = {}
        for (lat, lon), group in history.sort_values("time").groupby(["latitude", "longitude"]):
            satellite_lag_values[(float(lat), float(lon))] = group["satellite_pm25_aux"].tail(3).tolist()

        future_rows["satellite_pm25_aux"] = future_rows.apply(
            lambda row: satellite_lag_values[(float(row["latitude"]), float(row["longitude"]))][-1],
            axis=1,
        )
        future_rows["satellite_pm25_aux_lag_1"] = future_rows["satellite_pm25_aux"]
        future_rows["satellite_pm25_aux_lag_3"] = future_rows.apply(
            lambda row: float(np.mean(satellite_lag_values[(float(row["latitude"]), float(row["longitude"]))][-3:])),
            axis=1,
        )

    met_feature_columns = [
        "met_t2m_mean",
        "met_d2m_mean",
        "met_wind_speed_mean",
        "met_msl_mean",
        "met_sp_mean",
        "met_tcc_mean",
        "met_tp_sum",
    ]
    if any(column in feature_list for column in met_feature_columns):
        met_frame = load_meteorological_monthly_features()
        met_frame["time"] = pd.to_datetime(met_frame["time"]).dt.to_period("M").dt.to_timestamp()
        met_row = met_frame[met_frame["time"] == forecast_time]
        if met_row.empty:
            climatology = met_frame.assign(month=met_frame["time"].dt.month).groupby("month", as_index=False).mean(numeric_only=True)
            met_row = climatology[climatology["month"] == forecast_time.month]
        if not met_row.empty:
            for column in met_feature_columns:
                if column in met_row.columns:
                    future_rows[column] = float(met_row.iloc[0][column])

    previous_month = history["time"].max()
    previous_month_rows = history[history["time"] == previous_month].copy()
    future_rows = compute_spatial_lag_features(future_rows, previous_month_rows)

    fallback = future_rows["pm_lag_1"]
    future_rows["spatial_lag_mean_1"] = future_rows["spatial_lag_mean_1"].fillna(fallback)
    future_rows["spatial_lag_weighted_1"] = future_rows["spatial_lag_weighted_1"].fillna(fallback)
    future_rows["grid_density"] = future_rows["grid_density"].fillna(0)

    future_rows = pd.get_dummies(future_rows, columns=["season"], drop_first=False)

    for column in feature_list:
        if column not in future_rows.columns:
            future_rows[column] = 0

    return future_rows


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

    last_available = history["time"].max()
    if target_time <= last_available:
        raise ValueError(
            f"Requested month {target_time.strftime('%Y-%m')} is already inside the dataset. "
            f"Use `python -m src.predict --year {args.year} --month {args.month} --lat {args.lat} --lon {args.lon}` instead."
        )

    model = joblib.load(MODELS_DIR / "pm25_xgboost.joblib")
    feature_list = joblib.load(MODELS_DIR / "feature_columns.joblib")

    forecast_time = next_month(last_available)
    while forecast_time <= target_time:
        future_rows = build_future_month_rows(history, forecast_time, feature_list)
        future_rows["pm25"] = model.predict(future_rows[feature_list])
        history_columns = ["time", "latitude", "longitude", "pm25"]
        if "satellite_pm25_aux" in future_rows.columns:
            history_columns.append("satellite_pm25_aux")
        history = pd.concat([history, future_rows[history_columns]], ignore_index=True)
        forecast_time = next_month(forecast_time)

    target_rows = history[history["time"] == target_time].copy()
    target_rows["distance_to_query"] = ((target_rows["latitude"] - args.lat) ** 2 + (target_rows["longitude"] - args.lon) ** 2) ** 0.5
    row = target_rows.sort_values("distance_to_query").iloc[0]

    print("Future forecast summary")
    print(f"Last month in observed dataset: {last_available.strftime('%Y-%m')}")
    print(f"Forecast month: {target_time.strftime('%Y-%m')}")
    print(f"Requested location: lat={args.lat:.4f}, lon={args.lon:.4f}")
    print(f"Nearest grid cell: lat={row['latitude']:.4f}, lon={row['longitude']:.4f}")
    print(f"Forecast PM2.5: {float(row['pm25']):.3f}")

    if args.actual is not None:
        actual = float(args.actual)
        predicted = float(row["pm25"])
        absolute_error = abs(predicted - actual)
        squared_error = (predicted - actual) ** 2
        percentage_error = (absolute_error / actual * 100.0) if actual != 0 else float("nan")
        approximate_accuracy = (100.0 - percentage_error) if actual != 0 else float("nan")

        print(f"Actual PM2.5: {actual:.3f}")
        print(f"Absolute Error: {absolute_error:.3f}")
        print(f"Squared Error: {squared_error:.3f}")
        if actual != 0:
            print(f"Percentage Error: {percentage_error:.2f}%")
            print(f"Approximate Accuracy: {approximate_accuracy:.2f}%")
        else:
            print("Percentage Error: undefined for actual value 0")
            print("Approximate Accuracy: undefined for actual value 0")
    else:
        print("Approximate Accuracy: unavailable without --actual")


if __name__ == "__main__":
    main()
