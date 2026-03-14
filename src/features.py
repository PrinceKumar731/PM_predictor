from __future__ import annotations

import numpy as np
import pandas as pd


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


def _compute_spatial_features(data: pd.DataFrame, radius_degrees: float = 0.12, eps: float = 1e-6) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for current_time, month_frame in data.groupby("time", sort=True):
        coords = month_frame[["latitude", "longitude"]].to_numpy(dtype=float)
        values = month_frame["pm_lag_1"].to_numpy(dtype=float)

        spatial_mean = np.full(len(month_frame), np.nan, dtype=float)
        spatial_weighted = np.full(len(month_frame), np.nan, dtype=float)
        grid_density = np.zeros(len(month_frame), dtype=float)

        for idx, (lat, lon) in enumerate(coords):
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

        enriched = month_frame.copy()
        enriched["spatial_lag_mean_1"] = spatial_mean
        enriched["spatial_lag_weighted_1"] = spatial_weighted
        enriched["grid_density"] = grid_density
        frames.append(enriched)

    return pd.concat(frames, ignore_index=True)


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data = data.sort_values(["latitude", "longitude", "time"]).reset_index(drop=True)

    data["year"] = data["time"].dt.year
    data["month"] = data["time"].dt.month
    data["season"] = data["month"].map(month_to_season)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

    groups = data.groupby(["latitude", "longitude"], group_keys=False)
    data["pm_lag_1"] = groups["pm25"].shift(1)
    data["pm_lag_2"] = groups["pm25"].shift(2)
    data["pm_lag_3"] = groups["pm25"].shift(3)

    data["rolling_pm_3"] = groups["pm25"].transform(lambda series: series.shift(1).rolling(window=3, min_periods=1).mean())
    data["rolling_pm_6"] = groups["pm25"].transform(lambda series: series.shift(1).rolling(window=6, min_periods=1).mean())
    data["trend_1_3"] = (data["pm_lag_1"] - data["pm_lag_3"]) / 2.0

    first_time = data["time"].min()
    data["months_since_start"] = (data["time"].dt.year - first_time.year) * 12 + (data["time"].dt.month - first_time.month)

    data = _compute_spatial_features(data)

    fill_columns = [
        "pm_lag_1",
        "pm_lag_2",
        "pm_lag_3",
        "rolling_pm_3",
        "rolling_pm_6",
        "trend_1_3",
        "spatial_lag_mean_1",
        "spatial_lag_weighted_1",
    ]
    for column in fill_columns:
        data[column] = data[column].fillna(data["pm25"])

    data["grid_density"] = data["grid_density"].fillna(0)

    if "satellite_pm25_aux" in data.columns:
        satellite_groups = data.groupby(["latitude", "longitude"], group_keys=False)
        data["satellite_pm25_aux_lag_1"] = satellite_groups["satellite_pm25_aux"].shift(1)
        data["satellite_pm25_aux_lag_3"] = satellite_groups["satellite_pm25_aux"].transform(
            lambda series: series.shift(1).rolling(window=3, min_periods=1).mean()
        )
        data["satellite_pm25_aux_lag_1"] = data["satellite_pm25_aux_lag_1"].fillna(data["satellite_pm25_aux"])
        data["satellite_pm25_aux_lag_3"] = data["satellite_pm25_aux_lag_3"].fillna(data["satellite_pm25_aux"])

    met_columns = [
        "met_t2m_mean",
        "met_d2m_mean",
        "met_wind_speed_mean",
        "met_msl_mean",
        "met_sp_mean",
        "met_tcc_mean",
        "met_tp_sum",
    ]
    for column in met_columns:
        if column in data.columns:
            data[column] = data[column].interpolate(limit_direction="both")
            data[column] = data[column].fillna(data[column].median())

    data["season"] = data["season"].astype("category")
    model_data = pd.get_dummies(data, columns=["season"], drop_first=False)

    description = pd.DataFrame(
        [
            {"feature": "latitude", "reason": "Captures local north-south spatial differences inside Pune."},
            {"feature": "longitude", "reason": "Captures local east-west spatial differences inside Pune."},
            {"feature": "month", "reason": "Represents recurring monthly pollution seasonality."},
            {"feature": "year", "reason": "Captures slow structural changes across years."},
            {"feature": "month_sin", "reason": "Cyclic month encoding that preserves adjacency between December and January."},
            {"feature": "month_cos", "reason": "Second cyclic month component for smooth seasonal patterns."},
            {"feature": "months_since_start", "reason": "Linear time trend from the start of the record."},
            {"feature": "pm_lag_1", "reason": "Previous month PM2.5, usually the strongest persistence signal."},
            {"feature": "pm_lag_2", "reason": "Two-month memory for medium persistence."},
            {"feature": "pm_lag_3", "reason": "Three-month memory for longer temporal carryover."},
            {"feature": "rolling_pm_3", "reason": "Smoothed 3-month mean reduces short-term noise."},
            {"feature": "rolling_pm_6", "reason": "Smoothed 6-month mean captures sustained background loading."},
            {"feature": "trend_1_3", "reason": "Recent PM2.5 slope helps identify rising or falling trajectories."},
            {"feature": "spatial_lag_mean_1", "reason": "Average PM2.5 from neighboring cells in the previous month captures regional coherence."},
            {"feature": "spatial_lag_weighted_1", "reason": "Distance-weighted neighbor PM2.5 from the previous month preserves local plume structure."},
            {"feature": "grid_density", "reason": "Counts valid nearby neighbors and stabilizes spatial estimates."},
        ]
    )

    if "satellite_pm25_aux" in model_data.columns:
        satellite_description = pd.DataFrame(
            [
                {"feature": "satellite_pm25_aux", "reason": "Auxiliary monthly satellite-derived PM field used as an additional observational signal."},
                {"feature": "satellite_pm25_aux_lag_1", "reason": "Previous-month auxiliary satellite PM signal at the same location."},
                {"feature": "satellite_pm25_aux_lag_3", "reason": "Smoothed 3-month history of the auxiliary satellite PM signal."},
            ]
        )
        description = pd.concat([description, satellite_description], ignore_index=True)

    present_met_columns = [column for column in met_columns if column in model_data.columns]
    if present_met_columns:
        met_description = pd.DataFrame(
            [
                {"feature": "met_t2m_mean", "reason": "Monthly mean 2m temperature over Pune from the meteorological grid."},
                {"feature": "met_d2m_mean", "reason": "Monthly mean 2m dewpoint over Pune, capturing moisture conditions."},
                {"feature": "met_wind_speed_mean", "reason": "Monthly mean 10m wind speed over Pune, capturing dispersion strength."},
                {"feature": "met_msl_mean", "reason": "Monthly mean sea-level pressure, reflecting large-scale atmospheric conditions."},
                {"feature": "met_sp_mean", "reason": "Monthly mean surface pressure over Pune."},
                {"feature": "met_tcc_mean", "reason": "Monthly mean total cloud cover over Pune."},
                {"feature": "met_tp_sum", "reason": "Monthly total precipitation over Pune, capturing washout effects."},
            ]
        )
        met_description = met_description[met_description["feature"].isin(present_met_columns)]
        description = pd.concat([description, met_description], ignore_index=True)

    return model_data, description


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"time", "pm25", "source_file"}]
