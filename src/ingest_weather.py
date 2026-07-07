import os
import requests
import pandas as pd

def ingest_decade_weather_archive(output_path):
    """Fetches a full decade of hourly Houston weather data from Open-Meteo,

    incorporating advanced solar radiation and cloud cover metrics.
    """
    print("Initializing 10-Year Advanced Weather Extraction (2016-2025)...")
    
    # Houston geographic coordinates
    LAT, LON = 29.7604, -95.3698
    
    # Expanded timeline matching your ERCOT downloads
    START_DATE = "2016-01-01"
    END_DATE = "2025-12-31"
    
    # Base endpoint for Open-Meteo's historical archive
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    # ADVANCED PARAMETERS: Added cloud_cover and shortwave_radiation
    metrics = [
        "temperature_2m", "relative_humidity_2m", "apparent_temperature", 
        "precipitation", "wind_speed_10m", "cloud_cover", "shortwave_radiation"
    ]
    
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(metrics),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/Chicago"
    }
    
    try:
        print("Contacting Open-Meteo historical servers...")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Parse the hourly data arrays
        hourly_data = data["hourly"]
        
        # Standardize naming conversions into our local naming layout
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly_data["time"]),
            "temperature_f": hourly_data["temperature_2m"],
            "humidity_pct": hourly_data["relative_humidity_2m"],
            "apparent_temp_f": hourly_data["apparent_temperature"],
            "precipitation_in": hourly_data["precipitation"],
            "wind_speed_mph": hourly_data["wind_speed_10m"],
            "cloud_cover_pct": hourly_data["cloud_cover"],          # NEW ADVANCED FEATURE
            "solar_radiation_wm2": hourly_data["shortwave_radiation"] # NEW ADVANCED FEATURE
        })
        
        df.set_index("timestamp", inplace=True)
        
        # Write to our standard data path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path)
        print(f"✅ Success! 10-year weather matrix generated: {output_path}")
        print(df.tail(3))
        return df
        
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to fetch historical weather arrays. Details: {e}")
        return None

if __name__ == "__main__":
    target_destination = os.path.join("data", "raw", "houston_weather_2016_2025.csv")
    ingest_decade_weather_archive(output_path=target_destination)