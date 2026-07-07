ERCOT Houston Grid Load & Weather Forecasting Pipeline
An end-to-end machine learning and data engineering pipeline that ingests a full decade (2016–2025) of historical actual power grid load data and localized hourly weather data to forecast energy demand in the ERCOT Houston (Coast) region.

This project evaluates a baseline Ridge Regression model against an advanced LightGBM Gradient Boosting architecture using a robust, data-leakage-free walk-forward time-series validation strategy.

🏗️ System Architecture & Data Pipeline
The project decouples data ingestion, feature engineering, and model training into distinct, modular phases to ensure production maintainability:

Plaintext
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
🛠️ Feature Engineering Domain Mapping
To model the complex, non-linear relationships governing grid load, the feature engine transforms raw temporal and weather observations into highly predictive domain indicators:

Thermodynamic Response: Computes Cooling Degree Hours (CDH) and Heating Degree Hours (HDH) using a base human-comfort threshold of 65°F to capture the non-linear surge in HVAC demand during temperature extremes.

Solar & Cloud Interactions: Integrates Shortwave Radiation (W/m 
2
 ) and Cloud Cover Percentage to account for solar heat gain on buildings.

Calendar & Institutional Anomalies: Integrates the holidays engine to dynamically flag U.S. Federal Holidays across the decade, accounting for institutional power draw drops on weekdays like Thanksgiving or Christmas.

Structural Grid Growth Trend: Introduces a continuous time-elapsed index (grid_growth_trend) that empowers the machine learning trees to model and scale predictions against Houston's massive economic and population expansion over the 10-year horizon.

Autoregressive Thermal Inertia: Generates 24-hour and 168-hour (1-week) target lag horizons to lock onto grid momentum and cyclical weekly patterns.

🚦 Cross-Validation Strategy
Standard random K-Fold cross-validation introduces severe chronological data leakage in time-series forecasting. To ensure real-world viability, this pipeline enforces a Walk-Forward Time-Series Split (TimeSeriesSplit) across 4 sequential rolling folds:

Plaintext
Fold 1: Train [2016-01 to 2018-07] ➡️ Test [2018-07 to 2020-05]
Fold 2: Train [2016-01 to 2020-05] ➡️ Test [2020-05 to 2022-04] (Includes Winter Storm Uri)
Fold 3: Train [2016-01 to 2022-04] ➡️ Test [2022-04 to 2024-02]
Fold 4: Train [2016-01 to 2024-02] ➡️ Test [2024-02 to 2025-12]
📊 Experimental Results & Model Performance
The engineered features allow both models to capture the underlying structural patterns of the grid with exceptional fidelity:

Model Optimization	Mean Root Mean Squared Error (RMSE)	Mean Coeff. of Determination (R 
2
 )	Error Slashed
Ridge Regression Baseline	796.24 Megawatts	0.9210	Baseline
LightGBM Regressor (Champion)	752.74 Megawatts	0.9283	⬇️ 5.46%
Key Analytical Takeaways
Macro Explanatory Power: Achieving an R 
2
  of 0.9283 over a full decade demonstrates that the feature matrix successfully captures nearly 93% of the true variance in grid demand through varying economic climates and weather shifts.

Operational Economic Impact: Slashed the average forecasting error by 5.46% (saving over 43 MW of average hourly uncertainty). In real-world utility operations, this tighter error bound directly translates to reduced reliance on expensive, high-emission peaker plants.

Resiliency Testing (Fold 2 Anomaly): During the Fold 2 window (2020-05 to 2022-04), which contained the historic February 2021 Winter Storm Uri grid collapse, both models experienced lower R 
2
  scores (~0.88-0.89). Because actual load plummeted due to forced system blackouts while extreme weather inputs dictated record demand, the rigid linear baseline slightly outperformed LightGBM's complex decision trees, demonstrating a classic real-world overfitting challenge during unprecedented grid structural anomalies.

🚀 How to Execute the Pipeline
1. Environment Setup
Clone the repository and spin up your virtual environment:

Bash
python -m venv .venv
source .venv/Scripts/activate  # Or relevant OS command

# Install dependencies
pip install pandas numpy scikit-learn lightgbm openpyxl holidays requests
2. Sourcing Raw Data
Download the annual ERCOT Hourly Load Archive sheets (2016 through 2025) from the ERCOT Grid Hourly Load Archives.

Deposit the downloaded .xlsx/.xls files directly into:
data/ercot_historical_data_files/

3. Running the Pipeline End-to-End
Execute the scripts in order within your terminal:

Bash
# Ingest 10 years of hourly Houston weather data (Open-Meteo API)
python src/ingest_weather.py

# Parse and compile the 10 separate local ERCOT sheets into a clean raw matrix
python src/ingest_ercot.py

# Execute the feature engineering and transformation pipeline
python src/features.py

# Run the walk-forward time-series validation and training engine
python src/train.py