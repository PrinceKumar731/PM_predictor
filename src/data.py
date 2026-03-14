from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from .config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, METEOROLOGICAL_DIR, PM_RAW_DIR, SATELLITE_RAW_DIR


def find_netcdf_files(raw_dir: Path = PM_RAW_DIR) -> list[Path]:
    files = sorted(list(raw_dir.rglob("*.nc")))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {raw_dir}")
    return files


def _detect_coordinate_name(dataset: xr.Dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in dataset.coords or name in dataset.variables:
            return name
    raise KeyError(f"Could not find any coordinate name in {list(candidates)}")


def _detect_pm_variable(dataset: xr.Dataset) -> str:
    preferred = [
        "pm25",
        "PM25",
        "GWRPM25",
        "pm2_5",
        "PM2_5",
        "PM2.5",
        "annual",
    ]
    for name in preferred:
        if name in dataset.data_vars:
            return name

    for name, data_array in dataset.data_vars.items():
        lower_name = name.lower()
        if "pm" in lower_name and "25" in lower_name:
            return name
        long_name = str(data_array.attrs.get("long_name", "")).lower()
        if "pm2.5" in long_name or "pm25" in long_name:
            return name

    if len(dataset.data_vars) == 1:
        return next(iter(dataset.data_vars))

    raise KeyError(f"Unable to detect PM2.5 variable. Available variables: {list(dataset.data_vars)}")


def _subset_bbox(dataset: xr.Dataset) -> xr.Dataset:
    lat_name = _detect_coordinate_name(dataset, ["lat", "latitude", "Latitude", "LAT"])
    lon_name = _detect_coordinate_name(dataset, ["lon", "longitude", "Longitude", "LON"])

    lat_values = dataset[lat_name].values
    lon_values = dataset[lon_name].values

    lat_slice = slice(LAT_MIN, LAT_MAX) if lat_values[0] < lat_values[-1] else slice(LAT_MAX, LAT_MIN)
    lon_slice = slice(LON_MIN, LON_MAX) if lon_values[0] < lon_values[-1] else slice(LON_MAX, LON_MIN)

    return dataset.sel({lat_name: lat_slice, lon_name: lon_slice})


def _extract_time_value(frame: pd.DataFrame, source_path: Path, dataset: xr.Dataset) -> pd.Series:
    if "time" in frame.columns:
        return pd.to_datetime(frame["time"])

    if "month" in frame.columns:
        return pd.to_datetime(frame["month"])

    coverage = dataset.attrs.get("TIMECOVERAGE") or dataset.attrs.get("time_coverage") or dataset.attrs.get("TimeCoverage")
    if coverage:
        coverage_str = str(coverage).strip()
        try:
            return pd.Series(pd.to_datetime(coverage_str, format="%Y%m"), index=frame.index)
        except ValueError:
            pass

    match = re.search(r"((?:19|20)\d{2})(0[1-9]|1[0-2])", source_path.stem)
    if match:
        return pd.Series(pd.to_datetime(match.group(0), format="%Y%m"), index=frame.index)

    raise KeyError("No time information found in dataset metadata or filename.")


def _normalize_dataframe(frame: pd.DataFrame, pm_var: str, source_path: Path, dataset: xr.Dataset) -> pd.DataFrame:
    rename_map = {}
    for src, dst in {
        "lat": "latitude",
        "latitude": "latitude",
        "Latitude": "latitude",
        "lon": "longitude",
        "longitude": "longitude",
        "Longitude": "longitude",
        "time": "time",
        "month": "time",
        pm_var: "pm25",
    }.items():
        if src in frame.columns:
            rename_map[src] = dst

    frame = frame.rename(columns=rename_map)

    required = {"latitude", "longitude", "pm25"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns after normalization: {sorted(missing)}")

    frame["time"] = _extract_time_value(frame, source_path, dataset)
    frame["pm25"] = pd.to_numeric(frame["pm25"], errors="coerce")
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame = frame.dropna(subset=["time", "latitude", "longitude"])
    return frame[["time", "latitude", "longitude", "pm25"]]


def load_pm25_dataframe(files: list[Path] | None = None) -> pd.DataFrame:
    files = files or find_netcdf_files()
    frames: list[pd.DataFrame] = []

    for file_path in files:
        with xr.open_dataset(file_path) as dataset:
            dataset = _subset_bbox(dataset)
            pm_var = _detect_pm_variable(dataset)
            frame = dataset[[pm_var]].to_dataframe().reset_index()
            frame = _normalize_dataframe(frame, pm_var, file_path, dataset)
            frame["source_file"] = file_path.name
            frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["latitude", "longitude", "time"]).reset_index(drop=True)
    data = data.drop_duplicates(subset=["time", "latitude", "longitude"], keep="last")
    return data


def load_satellite_feature_dataframe(raw_dir: Path = SATELLITE_RAW_DIR) -> pd.DataFrame:
    files = find_netcdf_files(raw_dir)
    frames: list[pd.DataFrame] = []

    for file_path in files:
        with xr.open_dataset(file_path) as dataset:
            dataset = _subset_bbox(dataset)
            pm_var = _detect_pm_variable(dataset)
            frame = dataset[[pm_var]].to_dataframe().reset_index()
            frame = _normalize_dataframe(frame, pm_var, file_path, dataset)
            frame = frame.rename(columns={"pm25": "satellite_pm25_aux"})
            frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["latitude", "longitude", "time"]).reset_index(drop=True)
    data = data.drop_duplicates(subset=["time", "latitude", "longitude"], keep="last")
    return data


def handle_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data = data.sort_values(["latitude", "longitude", "time"]).reset_index(drop=True)

    data["pm25"] = (
        data.groupby(["latitude", "longitude"], group_keys=False)["pm25"]
        .apply(lambda series: series.interpolate(method="linear", limit_direction="both"))
    )

    spatial_month_fill = data.groupby("time")["pm25"].transform("median")
    global_fill = float(data["pm25"].median())
    data["pm25"] = data["pm25"].fillna(spatial_month_fill).fillna(global_fill)

    return data


def merge_satellite_feature(pm_frame: pd.DataFrame, satellite_frame: pd.DataFrame) -> pd.DataFrame:
    data = pm_frame.copy()
    satellite = satellite_frame.copy()

    for column in ["latitude", "longitude"]:
        data[column] = data[column].round(4)
        satellite[column] = satellite[column].round(4)

    merged = data.merge(satellite, on=["time", "latitude", "longitude"], how="left")

    if "satellite_pm25_aux" in merged.columns:
        merged["satellite_pm25_aux"] = (
            merged.groupby(["latitude", "longitude"], group_keys=False)["satellite_pm25_aux"]
            .apply(lambda series: series.interpolate(method="linear", limit_direction="both"))
        )
        merged["satellite_pm25_aux"] = merged["satellite_pm25_aux"].fillna(merged["pm25"])

    return merged


def load_meteorological_monthly_features(raw_dir: Path = METEOROLOGICAL_DIR) -> pd.DataFrame:
    instant_path = raw_dir / "data_stream-oper_stepType-instant.nc"
    accum_path = raw_dir / "data_stream-oper_stepType-accum.nc"
    if not instant_path.exists() or not accum_path.exists():
        raise FileNotFoundError("Meteorological NetCDF files not found.")

    with xr.open_dataset(instant_path) as instant_ds, xr.open_dataset(accum_path) as accum_ds:
        lat_name = _detect_coordinate_name(instant_ds, ["lat", "latitude", "Latitude", "LAT"])
        lon_name = _detect_coordinate_name(instant_ds, ["lon", "longitude", "Longitude", "LON"])
        time_name = _detect_coordinate_name(instant_ds, ["valid_time", "time", "month"])

        instant_subset = instant_ds.sel(
            {
                lat_name: slice(LAT_MAX, LAT_MIN) if instant_ds[lat_name].values[0] > instant_ds[lat_name].values[-1] else slice(LAT_MIN, LAT_MAX),
                lon_name: slice(LON_MIN, LON_MAX) if instant_ds[lon_name].values[0] < instant_ds[lon_name].values[-1] else slice(LON_MAX, LON_MIN),
            }
        )
        accum_subset = accum_ds.sel(
            {
                lat_name: slice(LAT_MAX, LAT_MIN) if accum_ds[lat_name].values[0] > accum_ds[lat_name].values[-1] else slice(LAT_MIN, LAT_MAX),
                lon_name: slice(LON_MIN, LON_MAX) if accum_ds[lon_name].values[0] < accum_ds[lon_name].values[-1] else slice(LON_MAX, LON_MIN),
            }
        )

        instant_frame = instant_subset[["u10", "v10", "d2m", "t2m", "msl", "sp", "tcc"]].to_dataframe().reset_index()
        accum_frame = accum_subset[["tp"]].to_dataframe().reset_index()

    instant_frame["time"] = pd.to_datetime(instant_frame[time_name]).dt.to_period("M").dt.to_timestamp()
    accum_frame["time"] = pd.to_datetime(accum_frame[time_name]).dt.to_period("M").dt.to_timestamp()

    instant_frame["wind_speed_10m"] = np.sqrt(instant_frame["u10"] ** 2 + instant_frame["v10"] ** 2)

    monthly_instant = (
        instant_frame.groupby("time", as_index=False)
        .agg(
            met_t2m_mean=("t2m", "mean"),
            met_d2m_mean=("d2m", "mean"),
            met_wind_speed_mean=("wind_speed_10m", "mean"),
            met_msl_mean=("msl", "mean"),
            met_sp_mean=("sp", "mean"),
            met_tcc_mean=("tcc", "mean"),
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    monthly_accum = (
        accum_frame.groupby("time", as_index=False)
        .agg(met_tp_sum=("tp", "sum"))
        .sort_values("time")
        .reset_index(drop=True)
    )

    return monthly_instant.merge(monthly_accum, on="time", how="outer").sort_values("time").reset_index(drop=True)


def merge_meteorological_features(pm_frame: pd.DataFrame, met_frame: pd.DataFrame) -> pd.DataFrame:
    data = pm_frame.copy()
    data["time"] = pd.to_datetime(data["time"]).dt.to_period("M").dt.to_timestamp()
    met = met_frame.copy()
    met["time"] = pd.to_datetime(met["time"]).dt.to_period("M").dt.to_timestamp()

    merged = data.merge(met, on="time", how="left")

    met_columns = [column for column in met.columns if column != "time"]
    for column in met_columns:
        merged[column] = merged[column].interpolate(limit_direction="both")
        merged[column] = merged[column].fillna(merged[column].median())

    return merged
