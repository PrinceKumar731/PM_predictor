# Asia Regional PM2.5 Prediction System

A high-resolution environmental monitoring and predictive dashboard powered by a proprietary machine learning pipeline. This system provides granular PM2.5 forecasting across the Asian continent, utilizing sophisticated spatial-temporal modeling and multi-source data integration.

## Demo video

https://github.com/user-attachments/assets/dcee66a4-435f-43c9-9a3f-50b582cd4307

## Project File Structure

```text
PM_predictor/
├── ml-service/               # Machine Learning Engine (Python)
│   ├── data/                 # Raw and processed datasets
│   │   ├── raw/              # NetCDF atmospheric and satellite files
│   │   └── processed/        # Engineered feature tables (Parquet)
│   ├── models/               # Serialized model artifacts (.joblib)
│   ├── src/                  # Core pipeline source code
│   │   ├── data.py           # NetCDF processing and normalization
│   │   ├── features.py       # Spatial-temporal feature engineering
│   │   ├── modeling.py       # XGBoost architecture and hyperparameter tuning
│   │   └── bulk_predict.py   # Batch processing engine
│   └── requirements.txt      # Scientific computing dependencies
├── backend/                  # Analytical Infrastructure (Node.js)
│   ├── index.js              # Express API and Python execution bridge
│   ├── package.json          # Backend dependencies
│   └── .env                  # Environment configuration
├── frontend/                 # Visualization Dashboard (React + TS)
│   ├── src/                  # UI components and state logic
│   │   ├── App.tsx           # Main dashboard orchestration
│   │   └── App.css           # Professional theme styles
│   └── package.json          # Frontend dependencies
└── README.md                 # Project documentation
```
### Stats For Nerds 
https://www.figma.com/make/rnWFCW62VYmGyRpMZFlSfy/Stats-for-nerds?p=f

## Model Architecture and Methodology

The predictive engine is built on a high-performance Gradient Boosting framework designed to handle the non-linear complexities of atmospheric chemistry.

### 1. Core Algorithm
- **XGBoost Regressor:** Optimized for sparse data handling and regularized boosting to prevent overfitting in complex regional climates.
- **Objective:** `reg:squarederror` with focused evaluation on RMSE and MAE.
- **Hyperparameters:** Depth-limited trees (max_depth: 6) with aggressive subsampling (0.85) to ensure generalization across diverse Asian topographies.

### 2. Feature Engineering Pipeline
The model utilizes a multi-dimensional feature vector:
- **Atmospheric Persistence:** Multi-horizon temporal lags (1-month, 3-month, and 6-month) to capture the "memory" of air masses.
- **Geospatial Correlation:** Distance-weighted spatial lag means calculated using a radial coordinate system to model pollutant dispersion from neighboring grid cells.
- **Meteorological Context:** Integration of mean 2m temperature, dew point, wind speed, surface pressure, cloud cover, and total precipitation.
- **Cyclic Seasonality:** Transformative encoding of monthly data into sine/cosine components to preserve the temporal continuity of seasonal shifts.

### 3. Interpretability Engine
- **SHAP Integration:** The system employs the Shapley Additive exPlanations framework to decompose individual predictions.
- **Feature Contribution:** Local explanations quantify the exact µg/m³ contribution of each factor (e.g., how much the current monsoon season is suppressing PM2.5 levels).
- **Global Importance:** Aggregated importance graphs provide a macro-level view of which environmental drivers dominate regional air quality.

<img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/a0992634-99e0-46f3-b5ec-698cf510e9a7" />


## System Architecture

The infrastructure ensures seamless integration between intensive ML computation and real-time user interaction.

### Analytical Pipeline
- **Inference Bridge:** The Node.js backend manages an asynchronous process execution pool, invoking Python inference scripts on-demand.
- **Regional Optimization:** Inference is optimized for Asia-wide coordinates, utilizing city-hub clustering to ensure high-accuracy sampling in inhabited regions.
- **Batch Processing:** A specialized bulk-analysis engine processes large-scale CSV datasets, applying regional accuracy bands to validate predictions against ground-truth data.

### Visualization Layer
- **Dual-Mode Mapping:** Real-time rendering of regional hotspots using Leaflet, supporting both discrete point analysis and continuous density-gradient visualizations.
- **Health Risk Quantization:** A proprietary mapping of PM2.5 concentrations to standardized health impact metrics, including daily and monthly cigarette-equivalent damage.

## Installation and Deployment

### Core Requirements
- Python 3.9+
- Node.js 16+
- Scientific Libraries: XGBoost, SHAP, Xarray, NetCDF4, Pandas

### Pipeline Setup
1. **Initialize ML Service:**
   ```bash
   cd ml-service
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Train Model:**
   ```bash
   python -m src.preprocess
   python -m src.train
   ```

### Dashboard Launch
1. **Start Backend:**
   ```bash
   cd backend
   npm install
   npm start
   ```
2. **Start Frontend:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Analytical Methodology

Prediction accuracy is validated using chronological time-series splitting (70/15/15) to ensure model robustness. Regional performance is monitored through automated accuracy bands, ensuring higher confidence intervals over densely inhabited Asian regions.

*Technical documentation and research artifacts are maintained within the respective service directories.*
