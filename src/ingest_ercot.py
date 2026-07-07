import os
import glob
import pandas as pd

def compile_local_ercot_decade(input_folder, output_path):
    """Scans the local download directory for ERCOT Excel files, normalizes their

    shifting structural layouts, isolates the Coast zone, and merges them.
    """
    print(f"Scanning directory for local ERCOT sheets: {input_folder}")
    
    # Use glob to find all excel files (.xlsx and .xls) inside your custom folder
    excel_files = glob.glob(os.path.join(input_folder, "*.xls*"))
    
    if not excel_files:
        raise FileNotFoundError(
            f"No Excel files found in {input_folder}. Ensure your downloads are dropped there!"
        )
        
    print(f"Found {len(excel_files)} historical files to process.")
    all_years_dfs = []
    
    for file_path in sorted(excel_files):
        file_name = os.path.basename(file_path)
        print(f"Processing: {file_name}...")
        
        try:
            # Read the excel file natively using openpyxl
            df = pd.read_excel(file_path, engine="openpyxl")
            
            # Clean and lower-case column headers to handle capitalization changes over the decade
            df.columns = [str(col).strip().lower() for col in df.columns]
            
            # 1. DYNAMICALLY FIND THE TIMESTAMP COLUMN
            # We look for common names ERCOT has used over the years
            possible_time_cols = ["hour ending", "hour_ending", "date", "timestamp"]
            time_col = None
            for col in df.columns:
                if col in possible_time_cols or "date" in col or "time" in col:
                    time_col = col
                    break
            
            if not time_col:
                # Fallback: if no name matches, assume it's the very first column
                time_col = df.columns[0]
                
            # 2. DYNAMICALLY FIND THE REGIONAL COSTA LOAD COLUMN
            # We explicitly need the 'coast' zone column for Houston
            if "coast" not in df.columns:
                print(f"  ⚠️ Warning: Could not find explicit 'coast' column in {file_name}. Skipping file.")
                continue
                
            # Extract just our target time column and the coast load column
            df_subset = df[[time_col, "coast"]].copy()
            df_subset.columns = ["timestamp", "coast_load_mw"]
            
            # 3. STANDARDIZE THE TIMESTAMPS
            # ERCOT times can sometimes be messy or read as strings; force conversion to datetimes
            df_subset["timestamp"] = pd.to_datetime(df_subset["timestamp"], errors="coerce")
            
            # Drop any rows where the timestamp couldn't be parsed properly
            df_subset.dropna(subset=["timestamp"], inplace=True)
            df_subset.set_index("timestamp", inplace=True)
            
            print(f"  Successfully extracted {len(df_subset)} hours from this sheet.")
            all_years_dfs.append(df_subset)
            
        except Exception as e:
            print(f"  ❌ Critical error parsing file {file_name}: {e}")
            continue

    if not all_years_dfs:
        print("CRITICAL: No datasets were successfully parsed. Pipeline halted.")
        return None
        
    # Combine all 10 years chronologically into a single seamless continuous timeline
    print("\nConsolidating all parsed sheets into a master timeline...")
    master_ercot_df = pd.concat(all_years_dfs).sort_index()
    
    # Drop any accidental duplicate hours that happen during data crossovers
    initial_count = len(master_ercot_df)
    master_ercot_df = master_ercot_df[~master_ercot_df.index.duplicated(keep="first")]
    print(f"Dropped {initial_count - len(master_ercot_df)} duplicate transition rows.")
    
    # Save our final clean actual data into the raw folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master_ercot_df.to_csv(output_path)
    
    print(f"✅ Success! Master 10-year grid load matrix saved to: {output_path}")
    print(master_ercot_df.head(3))
    print(master_ercot_df.tail(3))
    return master_ercot_df

if __name__ == "__main__":
    # Point directly to your custom folder path
    historical_folder = os.path.join("data", "ercot_historical_data_files")
    target_destination = os.path.join("data", "raw", "ercot_load_2016_2025.csv")
    
    compile_local_ercot_decade(input_folder=historical_folder, output_path=target_destination)