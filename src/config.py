from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PM_RAW_DIR = RAW_DIR / "Monthly"
SATELLITE_RAW_DIR = RAW_DIR / "satellite" / "Monthly"
METEOROLOGICAL_DIR = RAW_DIR / "meterological"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

LAT_MIN = 18.40
LAT_MAX = 18.70
LON_MIN = 73.70
LON_MAX = 74.10

TARGET_COLUMN = "pm25"
FEATURE_TABLE_NAME = "pune_pm25_features.parquet"
RAW_TABLE_NAME = "pune_pm25_raw.parquet"


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, BASE_DIR / "notebooks"]:
        path.mkdir(parents=True, exist_ok=True)
