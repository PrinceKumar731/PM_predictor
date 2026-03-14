from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from xgboost import XGBRegressor


@dataclass
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(frame: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> SplitData:
    unique_times = np.array(sorted(frame["time"].unique()))
    n_times = len(unique_times)

    train_end = max(1, int(n_times * train_frac))
    val_end = max(train_end + 1, int(n_times * (train_frac + val_frac)))
    val_end = min(val_end, n_times - 1) if n_times > 2 else n_times

    train_times = unique_times[:train_end]
    val_times = unique_times[train_end:val_end]
    test_times = unique_times[val_end:]

    train = frame[frame["time"].isin(train_times)].copy()
    validation = frame[frame["time"].isin(val_times)].copy()
    test = frame[frame["time"].isin(test_times)].copy()
    return SplitData(train=train, validation=validation, test=test)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    accuracy_pct = float(max(0.0, min(100.0, r2 * 100.0)))
    return {"rmse": rmse, "mae": mae, "r2": r2, "accuracy_pct": accuracy_pct}


def build_model(model_name: str = "xgboost", random_state: int = 42):
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )

    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric=["rmse", "mae"],
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=random_state,
        n_jobs=4,
    )


def tune_xgboost_with_time_cv(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    splitter = TimeSeriesSplit(n_splits=3)
    grid = ParameterGrid(
        {
            "n_estimators": [200, 350],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.08],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
        }
    )

    best_score = np.inf
    best_params: dict[str, float | int] | None = None

    for params in grid:
        fold_scores: list[float] = []
        for train_idx, val_idx in splitter.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=3,
                random_state=42,
                n_jobs=4,
                **params,
            )
            model.fit(X_train, y_train, verbose=False)
            pred = model.predict(X_val)
            fold_scores.append(np.sqrt(mean_squared_error(y_val, pred)))

        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    assert best_params is not None
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric=["rmse", "mae"],
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=42,
        n_jobs=4,
        **best_params,
    )
