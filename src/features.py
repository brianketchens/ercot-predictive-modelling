import os
import pandas as pd
import numpy as np
import holidays

def advanced_feature_engineering_pipeline():
    print("🛠️ Initiating Phase 2 Advanced Feature Engineering Engine...")
    
    # 1. Load the new expanded raw weather dataset
    weather_path = "data/raw/houston_weather_expanded.csv"
    if not os.path.exists(weather_path):
        raise FileNotFoundError(f"Missing required expanded weather file: {weather_path}")
    
    df = pd.read_csv(weather_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort chronologically to safeguard time-series window operations
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Extract Base Temporal Vectors
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # 3. ADVANCED METEOROLOGICAL TRANSFORMATIONS
    
    # A. Humidex / Realized Thermodynamic Load
    # Captures the compounding non-linear impact of moisture on cooling demand
    # Approximated formula for heat-sensation index using dew point
    e = 6.11 * np.exp(5417.7530 * ((1/273.16) - (1 / (273.15 + (df['dew_point_f'] - 32) * 5/9))))
    h = (5/9) * (e - 10.0)
    df['humidex'] = df['temp_f'] + h
    
    # B. Effective Solar Penetration Matrix
    # Scales Irradiance inversely with Cloud Cover to isolate building solar heat gain
    df['effective_solar_gain'] = df['direct_normal_irradiance_wm2'] * (1 - (df['cloud_cover_pct'] / 100))
    
    # C. Kinetic Wind Cooling Vector
    # Captures extreme wind-chill structural heat loss during winter anomalies
    df['kinetic_cooling_index'] = df['temp_f'] * (1 / (1 + np.log1p(df['wind_speed_mph'])))
    
    # D. Adaptive Thermal Adaptation Delta
    # Models consumer threshold adaptation relative to a rolling 7-day average
    df['rolling_7d_temp_mean'] = df['temp_f'].rolling(window=168, min_periods=1).mean()
    df['thermal_shock_delta'] = df['temp_f'] - df['rolling_7d_temp_mean']
    
    # E. Standard Baseline Thermodynamic Response (Base 65°F)
    df['cooling_degree_hours'] = (df['temp_f'] - 65).clip(lower=0)
    df['heating_degree_hours'] = (65 - df['temp_f']).clip(lower=0)
    
    # 4. CALENDAR ANOMALIES & DYNAMIC INSTITUTIONAL FLAGS
    us_holidays = holidays.US(years=range(2016, 2026))
    df['is_holiday'] = df['timestamp'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
    
    # Pre-holiday ramp down (flags the day prior to major baseline shifts)
    df['is_pre_holiday'] = df['is_holiday'].shift(-24).fillna(0).astype(int)
    
    # 5. STRUCTURAL GRID GROWTH TREND
    # Continuous index mapping long-term macro population/economic expansion
    df['grid_growth_trend'] = np.arange(len(df))
    
    # Save the expanded engineering matrix to processed directory
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "master_features_expanded.csv")
    
    df.to_csv(output_path, index=False)
    print(f"✅ Feature Engineering complete. Matrix Shape: {df.shape}")
    print(f"💾 Processed features saved directly to: {output_path}")

if __name__ == "__main__":
    advanced_feature_engineering_pipeline()