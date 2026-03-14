from __future__ import annotations

import joblib
import pandas as pd
from xgboost import XGBRegressor

from .config import FEATURE_TABLE_NAME, MODELS_DIR, PROCESSED_DIR, TARGET_COLUMN


def forecast_model_path(horizon: int) -> str:
    return str(MODELS_DIR / f"pm25_forecast_h{horizon}_xgboost.joblib")


def forecast_features_path(horizon: int) -> str:
    return str(MODELS_DIR / f"forecast_h{horizon}_feature_columns.joblib")


def future_safe_feature_columns(frame: pd.DataFrame) -> list[str]:
    include_prefixes = (
        "pm_lag_",
        "rolling_pm_",
        "trend_",
        "spatial_lag_",
        "grid_density",
        "month_sin",
        "month_cos",
        "months_since_start",
        "season_",
        "satellite_pm25_aux_lag_",
    )
    include_exact = {"latitude", "longitude", "month", "year"}

    features: list[str] = []
    for column in frame.columns:
        if column in {"time", TARGET_COLUMN, "source_file"}:
            continue
        if column in include_exact or any(column.startswith(prefix) for prefix in include_prefixes):
            features.append(column)
    return features


def build_forecast_training_frame(frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    data = frame.sort_values(["latitude", "longitude", "time"]).copy()
    data["target_next_pm25"] = data.groupby(["latitude", "longitude"])[TARGET_COLUMN].shift(-horizon)
    data["target_time"] = data.groupby(["latitude", "longitude"])["time"].shift(-horizon)
    data = data.dropna(subset=["target_next_pm25", "target_time"]).reset_index(drop=True)
    return data


def build_forecast_model(random_state: int = 42) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric=["rmse", "mae"],
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=2.0,
        min_child_weight=4,
        random_state=random_state,
        n_jobs=4,
    )


def save_forecast_artifacts(model: XGBRegressor, features: list[str], horizon: int) -> None:
    joblib.dump(model, forecast_model_path(horizon))
    joblib.dump(features, forecast_features_path(horizon))


def load_forecast_artifacts(horizon: int) -> tuple[XGBRegressor, list[str]]:
    model = joblib.load(forecast_model_path(horizon))
    features = joblib.load(forecast_features_path(horizon))
    return model, features


def load_feature_frame() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / FEATURE_TABLE_NAME).sort_values("time").reset_index(drop=True)
