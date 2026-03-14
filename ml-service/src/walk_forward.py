from __future__ import annotations

import argparse

import pandas as pd

from .config import FEATURE_TABLE_NAME, PROCESSED_DIR, TARGET_COLUMN, ensure_directories
from .features import feature_columns
from .modeling import build_model, evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward yearly validation for unseen future testing.")
    parser.add_argument("--start-year", type=int, default=2016, help="First test year to evaluate.")
    parser.add_argument("--end-year", type=int, default=None, help="Last test year to evaluate. Defaults to last available year.")
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()

    frame = pd.read_parquet(PROCESSED_DIR / FEATURE_TABLE_NAME).sort_values("time").reset_index(drop=True)
    features = feature_columns(frame)
    available_years = sorted(frame["year"].unique())
    end_year = args.end_year or int(available_years[-1])

    results: list[dict[str, float | int]] = []

    for test_year in available_years:
        if test_year < args.start_year or test_year > end_year:
            continue

        train_frame = frame[frame["year"] < test_year].copy()
        test_frame = frame[frame["year"] == test_year].copy()
        if train_frame.empty or test_frame.empty:
            continue

        model = build_model("xgboost", random_state=42)
        model.fit(train_frame[features], train_frame[TARGET_COLUMN], verbose=False)
        pred = model.predict(test_frame[features])
        metrics = evaluate_predictions(test_frame[TARGET_COLUMN], pred)
        metrics["year"] = int(test_year)
        results.append(metrics)

    results_frame = pd.DataFrame(results)
    output_path = PROCESSED_DIR / "walk_forward_results.csv"
    results_frame.to_csv(output_path, index=False)

    for _, row in results_frame.iterrows():
        print(
            f"{int(row['year'])}: RMSE={row['rmse']:.2f}, "
            f"MAE={row['mae']:.2f}, R2={row['r2']:.3f}, "
            f"Accuracy={row['accuracy_pct']:.2f}%"
        )

    if not results_frame.empty:
        print(
            f"AVERAGE: RMSE={results_frame['rmse'].mean():.2f}, "
            f"MAE={results_frame['mae'].mean():.2f}, "
            f"R2={results_frame['r2'].mean():.3f}, "
            f"Accuracy={results_frame['accuracy_pct'].mean():.2f}%"
        )
        print(f"Saved walk-forward results to {output_path}")


if __name__ == "__main__":
    main()
