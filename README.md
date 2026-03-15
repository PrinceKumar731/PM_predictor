# Asia Regional PM2.5 Prediction System

A high-resolution environmental monitoring and predictive dashboard powered by a proprietary machine learning pipeline. This system provides granular PM2.5 forecasting across the Asian continent, utilizing sophisticated spatial-temporal modeling and multi-source data integration.

## System Architecture

The project is structured as a full-stack monorepo, integrating high-performance data processing with a modern visualization interface.

### 1. Machine Learning Engine (ml-service)
The core of the system is a Python-based pipeline designed for high-resolution atmospheric modeling.
- **Model Architecture:** Employs an optimized XGBoost gradient boosting regressor, tuned specifically for atmospheric persistence and regional variance.
- **Data Integration:** Processes primary PM2.5 archives alongside auxiliary satellite-derived fields and multi-variable meteorological grids (Temperature, Dew Point, Wind Speed, Surface Pressure, Cloud Cover, and Precipitation).
- **Feature Engineering:**
    - **Temporal:** Cyclic seasonal encoding, multi-horizon temporal lags (1m, 3m, 6m), and long-term trend analysis.
    - **Spatial:** Neighboring grid-cell synchronization using distance-weighted spatial lag means to capture regional pollutant drift.
- **Interpretability:** Integrated SHAP (SHapley Additive exPlanations) engine to quantify the contribution of each environmental factor to the final prediction.

### 2. Analytical Backend (backend)
A Node.js infrastructure that serves as the interface between the modeling engine and the dashboard.
- **Inference Pipeline:** Executes the predictive models on-demand to generate real-time atmospheric estimates for any requested geographical coordinate.
- **Batch Processing:** A specialized bulk-analysis engine capable of processing large-scale CSV datasets for regional air quality audits and accuracy assessment.
- **Geospatial Intelligence:** Dynamic coordinate retrieval for major Asian population centers to ensure sensible spatial sampling.

### 3. Visualization Dashboard (frontend)
A professional React + TypeScript interface designed for environmental researchers and policy makers.
- **Predictor Dashboard:** Detailed view of predicted PM2.5 levels with a prioritized feature importance graph.
- **Asia Heatmap:** A dual-mode spatial visualization offering both discrete point-based analysis and continuous density-gradient heatmaps.
- **Health Impact Metrics:** Proprietary algorithm to translate particulate concentration into standardized health risk equivalents (standardized cigarette-impact metrics).
- **Bulk Analysis Interface:** Dedicated secure portal for uploading regional data files and downloading generated analytical reports.

## Installation and Deployment

### Core Requirements
- Python 3.9+
- Node.js 16+
- Scientific Computing Libraries: XGBoost, SHAP, Xarray, NetCDF4, Pandas

### Modeling Pipeline Setup
1. Navigate to the machine learning directory:
   ```bash
   cd ml-service
   ```
2. Initialize the virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Prepare the model artifacts:
   ```bash
   python -m src.preprocess
   python -m src.train
   ```

### Dashboard Deployment
1. Start the analytical backend:
   ```bash
   cd backend
   npm install
   npm start
   ```
2. Launch the visualization interface:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Analytical Methodology

The system adheres to a rigorous environmental data science framework. Prediction accuracy is validated using chronological time-series splitting to ensure model robustness against future atmospheric shifts. Regional performance is monitored using automated accuracy bands, ensuring higher confidence intervals over densely inhabited Asian regions.

*Technical documentation and research artifacts are maintained within the respective service directories.*
