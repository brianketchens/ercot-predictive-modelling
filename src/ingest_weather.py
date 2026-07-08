import os
import time
import requests
import pandas as pd

def fetch_houston_weather_pipeline():
    """
    Ingests 10 years of granular meteorological data for the Houston Coast region
    via the Open-Meteo Archive API, expanding features to include thermodynamic,
    solar irradiance, and atmospheric kinetic vectors.
    """
    print("🚀 Initiating Open-Meteo Historical Ingestion Pipeline (2016-2025)...")
    
    # Houston central coordinates
    LATITUDE = 29.7604
    LONGITUDE = -95.3698
    
    # Target expanded meteorological parameters
    variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "cloud_cover",
        "wind_speed_10m",
        "direct_normal_irradiance"
    ]
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": "2016-01-01",
        "end_date": "2025-12-31",
        "hourly": ",".join(variables),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/Chicago"
    }
    
    # Robust API request execution with fallback retry logic
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Connection attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                raise
            time.sleep(5)
            
    # Extract and parse hourly data array
    hourly_data = data["hourly"]
    
    # Build core dataframe matrix
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly_data["time"]),
        "temp_f": hourly_data["temperature_2m"],
        "humidity_pct": hourly_data["relative_humidity_2m"],
        "dew_point_f": hourly_data["dew_point_2m"],
        "apparent_temp_f": hourly_data["apparent_temperature"],
        "cloud_cover_pct": hourly_data["cloud_cover"],
        "wind_speed_mph": hourly_data["wind_speed_10m"],
        "direct_normal_irradiance_wm2": hourly_data["direct_normal_irradiance"]
    })
    
    # Ensure raw output directory exists
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "houston_weather_expanded.csv")
    df.to_csv(output_path, index=False)
    
    print(f"✅ Ingestion complete. Shape: {df.shape}")
    print(f"💾 Expanded raw weather saved directly to: {output_path}")

if __name__ == "__main__":
    fetch_houston_weather_pipeline()