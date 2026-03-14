from __future__ import annotations

import argparse

import joblib
import pandas as pd
import shap

from .config import FEATURE_TABLE_NAME, MODELS_DIR, PROCESSED_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict PM2.5 for a Pune location and month.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=5, help="Number of top SHAP factors to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(PROCESSED_DIR / FEATURE_TABLE_NAME)
    model = joblib.load(MODELS_DIR / "pm25_xgboost.joblib")
    features = joblib.load(MODELS_DIR / "feature_columns.joblib")

    subset = frame[(frame["year"] == args.year) & (frame["month"] == args.month)].copy()
    if subset.empty:
        raise ValueError("Requested year/month not found in processed feature table.")

    subset["distance_to_query"] = ((subset["latitude"] - args.lat) ** 2 + (subset["longitude"] - args.lon) ** 2) ** 0.5
    row = subset.sort_values("distance_to_query").iloc[0]

    predicted_pm25 = float(model.predict(pd.DataFrame([row[features]]))[0])
    actual_pm25 = float(row["pm25"])
    absolute_error = abs(predicted_pm25 - actual_pm25)
    percentage_error = (absolute_error / actual_pm25 * 100.0) if actual_pm25 != 0 else float("nan")
    approximate_accuracy = (100.0 - percentage_error) if actual_pm25 != 0 else float("nan")

    sample = pd.DataFrame([row[features]])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    explanation = pd.DataFrame(
        {
            "feature": features,
            "feature_value": sample.iloc[0].values,
            "shap_value": shap_values[0],
        }
    )
    explanation["abs_shap"] = explanation["shap_value"].abs()
    explanation = explanation.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    print("Prediction summary")
    print(f"Requested location: lat={args.lat:.4f}, lon={args.lon:.4f}")
    print(f"Nearest grid cell: lat={row['latitude']:.4f}, lon={row['longitude']:.4f}")
    print(f"Requested month: {args.year}-{args.month:02d}")
    print(f"Predicted PM2.5: {predicted_pm25:.3f}")
    print(f"Actual PM2.5 in dataset: {actual_pm25:.3f}")
    print(f"Approximate Accuracy: {approximate_accuracy:.2f}%")
    print("Top factors affecting the prediction:")
    for _, item in explanation.head(args.top_k).iterrows():
        direction = "increased" if item["shap_value"] >= 0 else "decreased"
        print(
            f"- {item['feature']}: value={item['feature_value']:.6g}, "
            f"SHAP={item['shap_value']:.3f} ({direction} prediction)"
        )

    met_explanation = explanation[explanation["feature"].str.startswith("met_")].copy()
    if not met_explanation.empty:
        print("Meteorological factors affecting the prediction:")
        for _, item in met_explanation.sort_values("abs_shap", ascending=False).iterrows():
            direction = "increased" if item["shap_value"] >= 0 else "decreased"
            print(
                f"- {item['feature']}: value={item['feature_value']:.6g}, "
                f"SHAP={item['shap_value']:.3f} ({direction} prediction)"
            )


if __name__ == "__main__":
    main()
