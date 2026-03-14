from __future__ import annotations

import json

import joblib
import pandas as pd

from .config import FEATURE_TABLE_NAME, MODELS_DIR, PLOTS_DIR, PROCESSED_DIR, TARGET_COLUMN, ensure_directories
from .features import feature_columns
from .modeling import build_model, chronological_split, evaluate_predictions, tune_xgboost_with_time_cv


def main() -> None:
    ensure_directories()
    feature_path = PROCESSED_DIR / FEATURE_TABLE_NAME
    frame = pd.read_parquet(feature_path).sort_values("time").reset_index(drop=True)

    splits = chronological_split(frame)
    features = feature_columns(frame)

    X_train = splits.train[features]
    y_train = splits.train[TARGET_COLUMN]
    X_val = splits.validation[features]
    y_val = splits.validation[TARGET_COLUMN]
    X_test = splits.test[features]
    y_test = splits.test[TARGET_COLUMN]

    if len(splits.train["time"].unique()) >= 6:
        model = tune_xgboost_with_time_cv(X_train, y_train)
    else:
        model = build_model("xgboost")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = evaluate_predictions(y_val, val_pred)
    test_metrics = evaluate_predictions(y_test, test_pred)

    joblib.dump(model, MODELS_DIR / "pm25_xgboost.joblib")
    joblib.dump(features, MODELS_DIR / "feature_columns.joblib")
    joblib.dump(
        {
            "validation": val_metrics,
            "test": test_metrics,
        },
        MODELS_DIR / "metrics.joblib",
    )
    with open(MODELS_DIR / "training_history.json", "w", encoding="utf-8") as file:
        json.dump(model.evals_result(), file, indent=2)

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(PLOTS_DIR / "feature_importance.csv", index=False)

    print("Validation metrics:", val_metrics)
    print(f"Validation accuracy: {val_metrics['accuracy_pct']:.2f}%")
    print("Test metrics:", test_metrics)
    print(f"Test accuracy: {test_metrics['accuracy_pct']:.2f}%")
    print(f"Saved model to {MODELS_DIR / 'pm25_xgboost.joblib'}")


if __name__ == "__main__":
    main()
