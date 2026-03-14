from __future__ import annotations

from .config import FEATURE_TABLE_NAME, PROCESSED_DIR, RAW_TABLE_NAME, ensure_directories
from .data import (
    handle_missing_values,
    load_meteorological_monthly_features,
    load_pm25_dataframe,
    load_satellite_feature_dataframe,
    merge_meteorological_features,
    merge_satellite_feature,
)
from .features import build_features


def main() -> None:
    ensure_directories()

    raw_df = load_pm25_dataframe()
    clean_df = handle_missing_values(raw_df)
    try:
        satellite_df = load_satellite_feature_dataframe()
        clean_df = merge_satellite_feature(clean_df, satellite_df)
    except FileNotFoundError:
        pass
    try:
        met_df = load_meteorological_monthly_features()
        clean_df = merge_meteorological_features(clean_df, met_df)
    except FileNotFoundError:
        pass
    feature_df, feature_description = build_features(clean_df)

    raw_path = PROCESSED_DIR / RAW_TABLE_NAME
    feature_path = PROCESSED_DIR / FEATURE_TABLE_NAME
    description_path = PROCESSED_DIR / "feature_description.csv"

    clean_df.to_parquet(raw_path, index=False)
    feature_df.to_parquet(feature_path, index=False)
    feature_description.to_csv(description_path, index=False)

    print(f"Saved cleaned data to {raw_path}")
    print(f"Saved feature table to {feature_path}")
    print(f"Saved feature descriptions to {description_path}")
    print(f"Rows: {len(feature_df):,}")
    print(f"Date range: {feature_df['time'].min().date()} to {feature_df['time'].max().date()}")


if __name__ == "__main__":
    main()
