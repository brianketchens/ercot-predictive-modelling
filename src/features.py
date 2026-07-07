import os
import pandas as pd
import numpy as np
import holidays

def build_advanced_feature_pipeline(weather_path, ercot_path, output_path):
    """Merges a decade of weather and load matrices, and engineers advanced

    features including holidays, solar indexes, and structural time-trends.
    """
    print("Initializing 10-Year Advanced Feature Engineering Pipeline...")
    
    # 1. LOAD RAW DATASETS
    if not os.path.exists(weather_path) or not os.path.exists(ercot_path):
        raise FileNotFoundError("Missing raw decade datasets. Ensure Steps 1 and 2 ran successfully!")
        
    df_weather = pd.read_csv(weather_path, parse_dates=["timestamp"], index_col="timestamp")
    df_ercot = pd.read_csv(ercot_path, parse_dates=["timestamp"], index_col="timestamp")
    
    # 2. SEAMLESS TIME-SERIES MERGE
    print("Aligning weather metrics with grid load timelines...")
    # Inner join ensures we only keep hours where BOTH datasets have records
    df = df_ercot.join(df_weather, how="inner")
    print(f"  Synchronized matrix contains {len(df)} total operational hours.")
    
    # 3. ADVANCED CALENDAR & HOLIDAY ENGINEERING
    print("Building temporal features and federal holiday mappings...")
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Initialize the automated US holiday lookups spanning our decade
    us_holidays = holidays.US(years=range(2016, 2027))
    # Map each timestamp's exact date to a binary 1 or 0 flag
    df["is_holiday"] = [1 if str(date.date()) in us_holidays else 0 for date in df.index]
    
    # 4. DOMAIN THERMODYNAMIC ENGINEERING (Heating/Cooling Degree Hours)
    # Energy demand responds non-linearly to temperature. 
    # Human comfort baseline sits around 65°F. 
    df["cooling_degree_hours"] = (df["temperature_f"] - 65).clip(lower=0)
    df["heating_degree_hours"] = (65 - df["temperature_f"]).clip(lower=0)
    
    # 5. INDUSTRIAL TIME-TREND (Economic / Population Growth Factor)
    # Creates a linear increment column counting up from hour 0 to the final hour.
    # This acts as an architectural hook allowing LightGBM to scale predictions upward over time.
    df["grid_growth_trend"] = np.arange(len(df))
    
    # 6. TIME-LAGGED BACKTRACKS (Captures thermal inertia / momentum)
    print("Calculating historical load lag signatures...")
    # Tells the model what the grid load was exactly 24 hours ago and 1 week ago
    df["load_lag_24h"] = df["coast_load_mw"].shift(24)
    df["load_lag_168h"] = df["coast_load_mw"].shift(168)
    
    # 7. CLEANUP & SAVE
    # Drop rows at the very beginning of 2016 that don't have lag values yet
    initial_len = len(df)
    df.dropna(inplace=True)
    print(f"  Dropped {initial_len - len(df)} initialization rows due to lag horizons.")
    
    # Save into the processed directory warehouse
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"✅ Success! Advanced feature matrix generated: {output_path}")
    print("\nAvailable features for your Machine Learning model:")
    print(list(df.columns))
    return df

if __name__ == "__main__":
    weather_input = os.path.join("data", "raw", "houston_weather_2016_2025.csv")
    ercot_input = os.path.join("data", "raw", "ercot_load_2016_2025.csv")
    processed_output = os.path.join("data", "processed", "engineered_energy_features.csv")
    
    build_advanced_feature_pipeline(weather_input, ercot_input, processed_output)