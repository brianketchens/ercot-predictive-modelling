# ERCOT Houston Grid Load & Weather Forecasting Pipeline

An end-to-end machine learning and data engineering pipeline that ingests a full decade (2016–2025) of historical grid load data and localized hourly weather data to forecast energy demand in the ERCOT Houston (Coast) region.

This project evaluates a baseline Ridge Regression model against an advanced LightGBM gradient boosting architecture using a robust, leakage-free walk-forward time-series validation strategy.

---

## 🏗️ System Architecture & Data Pipeline

The project decouples data ingestion, feature engineering, and model training into distinct, modular phases to ensure production maintainability:

```text
ercot-energy-project/
├── data/
│   ├── ercot_historical_data_files/   # Raw manual ERCOT .xlsx downloads (2016-2025)
│   ├── raw/                           # Standardized raw CSV outputs from ingestion scripts
│   └── processed/                     # Synchronized master feature matrix
├── src/
│   ├── ingest_weather.py              # Requests 10-year Open-Meteo meteorological archive
│   ├── ingest_ercot.py                # Compiles and normalizes shifting historical ERCOT sheets
│   ├── features.py                    # Multi-variable feature engineering pipeline
│   └── train.py                       # Walk-forward time-series training loop
└── README.md
```

---

## 🛠️ Feature Engineering Domain Mapping

To model the complex, non-linear relationships governing grid load, the feature engine transforms raw temporal and weather observations into highly predictive domain indicators:

- **Thermodynamic Response:** Computes Cooling Degree Hours (CDH) and Heating Degree Hours (HDH) using a base human-comfort threshold of 65 °F to capture the non-linear surge in HVAC demand during temperature extremes.
- **Solar & Cloud Interactions:** Integrates shortwave radiation (W/m²) and cloud cover percentage to account for solar heat gain on buildings.
- **Calendar & Institutional Anomalies:** Uses the `holidays` engine to dynamically flag U.S. federal holidays across the decade, accounting for institutional power draw drops on weekdays such as Thanksgiving or Christmas.
- **Structural Grid Growth Trend:** Introduces a continuous time-elapsed index (`grid_growth_trend`) that allows the tree-based models to scale predictions against Houston's economic and population expansion over the 10-year horizon.
- **Autoregressive Thermal Inertia:** Generates 24-hour and 168-hour (1-week) target lag horizons to capture grid momentum and cyclical weekly patterns.

---

## 🚦 Cross-Validation Strategy

Standard random K-Fold cross-validation introduces severe chronological data leakage in time-series forecasting. To ensure real-world viability, this pipeline enforces a walk-forward time-series split (`TimeSeriesSplit`) across four sequential rolling folds:

| Fold | Training Window | Test Window | Notes |
| :--- | :--- | :--- | :--- |
| 1 | 2016-01 → 2018-07 | 2018-07 → 2020-05 | |
| 2 | 2016-01 → 2020-05 | 2020-05 → 2022-04 | Includes Winter Storm Uri |
| 3 | 2016-01 → 2022-04 | 2022-04 → 2024-02 | |
| 4 | 2016-01 → 2024-02 | 2024-02 → 2025-12 | |

---

## 📊 Experimental Results & Model Performance

The engineered features allow both models to capture the underlying structural patterns of the grid with high fidelity:

| Model | Mean RMSE (MW) | Mean R² | Error Reduction |
| :--- | :--- | :--- | :--- |
| Ridge Regression (baseline) | 796.24 | 0.9210 | — |
| **LightGBM Regressor (champion)** | **752.74** | **0.9283** | ⬇️ 5.46% |

---

## 🔑 Key Analytical Takeaways

- **Macro explanatory power:** An R² of 0.9283 across a full decade demonstrates that the feature matrix captures nearly 93% of the true variance in grid demand through varying economic climates and weather shifts.
- **Operational economic impact:** Reducing average forecasting error by 5.46% eliminates roughly 43 MW of average hourly uncertainty. In utility operations, a tighter error bound translates directly into reduced reliance on expensive, high-emission peaker plants.
- **Resiliency testing (Fold 2 anomaly):** The Fold 2 test window (2020-05 → 2022-04) contains the February 2021 Winter Storm Uri grid collapse, and both models posted lower R² scores (~0.88–0.89). Because actual load plummeted under forced blackouts while extreme weather inputs implied record demand, the rigid linear baseline slightly outperformed LightGBM's decision trees — a classic illustration of tree-based overfitting during unprecedented structural anomalies.

---

## 🚀 How to Execute the Pipeline

### 1. Environment setup

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm openpyxl holidays requests
```

### 2. Source the raw data

Download the annual ERCOT hourly load archive sheets (2016 through 2025) from the ERCOT Grid Hourly Load Archives, then place the `.xlsx` / `.xls` files directly into:

```text
data/ercot_historical_data_files/
```

### 3. Run the pipeline end to end

Execute the scripts in order:

```bash
# Ingest 10 years of hourly Houston weather data (Open-Meteo API)
python src/ingest_weather.py

# Parse and compile the 10 separate local ERCOT sheets into a clean raw matrix
python src/ingest_ercot.py

# Execute the feature engineering and transformation pipeline
python src/features.py

# Run the walk-forward time-series validation and training engine
python src/train.py
```
