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
    e = 6.11 * np.exp(5417.7530 * ((1/273.16) - (1 / (273.15 + (df['dew_point_f'] - 32) * 5/9))))
    h = (5/9) * (e - 10.0)
    df['humidex'] = df['temp_f'] + h
    
    # B. Effective Solar Penetration Matrix
    df['effective_solar_gain'] = df['direct_normal_irradiance_wm2'] * (1 - (df['cloud_cover_pct'] / 100))
    
    # C. Kinetic Wind Cooling Vector
    df['kinetic_cooling_index'] = df['temp_f'] * (1 / (1 + np.log1p(df['wind_speed_mph'])))
    
    # D. Adaptive Thermal Adaptation Delta
    df['rolling_7d_temp_mean'] = df['temp_f'].rolling(window=168, min_periods=1).mean()
    df['thermal_shock_delta'] = df['temp_f'] - df['rolling_7d_temp_mean']
    
    # E. Standard Baseline Thermodynamic Response (Base 65°F)
    df['cooling_degree_hours'] = (df['temp_f'] - 65).clip(lower=0)
    df['heating_degree_hours'] = (65 - df['temp_f']).clip(lower=0)
    
    # 4. CALENDAR ANOMALIES & DYNAMIC INSTITUTIONAL FLAGS
    us_holidays = holidays.US(years=range(2016, 2026))
    df['is_holiday'] = df['timestamp'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
    
    # Pre-holiday ramp down
    df['is_pre_holiday'] = df['is_holiday'].shift(-24).fillna(0).astype(int)
    
    # 5. STRUCTURAL GRID GROWTH TREND
    df['grid_growth_trend'] = np.arange(len(df))
    
    # === PHASE 2 INNER JOIN WITH ERCOT ACTUAL TARGETS ===
    ercot_target_path = "data/raw/ercot_load_2016_2025.csv"
    if not os.path.exists(ercot_target_path):
        print(f"⚠️ Target load file not found at {ercot_target_path}. Please run 'python src/ingest_ercot.py' first!")
        master_df = df
    else:
        print("🔗 Synchronizing expanded feature matrix with actual grid load targets...")
        ercot_df = pd.read_csv(ercot_target_path)
        ercot_df['timestamp'] = pd.to_datetime(ercot_df['timestamp'])
        
        # Merge weather features and actual load targets on the timestamp index
        master_df = pd.merge(df, ercot_df, on='timestamp', how='inner')
    
    # Save the consolidated engineering matrix to processed directory
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "master_features_expanded.csv")
    
    master_df.to_csv(output_path, index=False)
    print(f"✅ Feature Engineering complete. Final Master Matrix Shape: {master_df.shape}")
    print(f"💾 Processed master dataset saved to: {output_path}")

if __name__ == "__main__":
    advanced_feature_engineering_pipeline()