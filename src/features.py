import os
import pandas as pd
import numpy as np
import holidays

def advanced_feature_engineering_pipeline():
    print("🛠️ Re-assembling Full Phase 2 Feature Engineering Matrix (22 Features)...")
    
    weather_path = "data/raw/houston_weather_expanded.csv"
    if not os.path.exists(weather_path):
        raise FileNotFoundError(f"Missing required expanded weather file: {weather_path}")
    
    df = pd.read_csv(weather_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. Base Temporal Vectors
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # 2. Complete Meteorological Transformations (All 10 Restored)
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
    
    # E. Degree Hours
    df['cooling_degree_hours'] = (df['temp_f'] - 65).clip(lower=0)
    df['heating_degree_hours'] = (65 - df['temp_f']).clip(lower=0)
    
    # F. Peak Demand Hour Interaction Focus
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if 15 <= x <= 19 else 0)
    df['peak_thermal_stress'] = df['cooling_degree_hours'] * df['is_peak_hour']
    df = df.drop(columns=['is_peak_hour'])
    
    # 3. Calendar Flags
    us_holidays = holidays.US(years=range(2016, 2026))
    df['is_holiday'] = df['timestamp'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
    df['is_pre_holiday'] = df['is_holiday'].shift(-24).fillna(0).astype(int)
    
    # 4. Phase 2 Sync with Actual Load Targets
    ercot_target_path = "data/raw/ercot_load_2016_2025.csv"
    if not os.path.exists(ercot_target_path):
         raise FileNotFoundError(f"Missing target load file at {ercot_target_path}")
         
    print("🔗 Synchronizing all 22 features with grid load targets...")
    ercot_df = pd.read_csv(ercot_target_path)
    ercot_df['timestamp'] = pd.to_datetime(ercot_df['timestamp'])
    
    # Merge everything seamlessly on timestamp
    master_df = pd.merge(df, ercot_df, on='timestamp', how='inner')
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "master_features_expanded.csv")
    
    master_df.to_csv(output_path, index=False)
    print(f"✅ Feature Engineering complete. Matrix Shape: {master_df.shape}")

if __name__ == "__main__":
    advanced_feature_engineering_pipeline()