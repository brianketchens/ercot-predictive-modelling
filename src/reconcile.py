import os
import pandas as pd
import numpy as np
import lightgbm as lgb

def run_production_reconciliation_pipeline():
    print("🏋️ Loading Matrix for Production Two-Model HTS Pipeline...")
    processed_matrix = os.path.join("data", "processed", "master_features_expanded.csv")
    
    df = pd.read_csv(processed_matrix, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    
    # Define our strict chronological train/test split (Fold 4 boundary)
    split_date = pd.to_datetime("2024-02-01")
    train_df = df[df["timestamp"] < split_date].reset_index(drop=True)
    test_df = df[df["timestamp"] >= split_date].reset_index(drop=True)
    
    # =================================================================
    # 🧠 PHASE 1: AGGREGATE DATA & TRAIN MODEL A (MACRO DAILY PEAK MODEL)
    # =================================================================
    print("📈 Engineering Macro Daily Aggregates for Model A...")
    
    # Create daily features out of our hourly data for training
    def build_daily_features(dataframe):
        daily = dataframe.groupby("date").agg(
            actual_daily_peak=("coast_load_mw", "max"),
            max_temp=("temp_f", "max"),
            mean_temp=("temp_f", "mean"),
            max_humidex=("humidex", "max"),
            day_of_week=("day_of_week", "first"),
            day_of_year=("day_of_year", "first"),
            year=("year", "first")
        ).reset_index()
        return daily

    daily_train = build_daily_features(train_df)
    daily_test = build_daily_features(test_df)
    
    macro_features = ["max_temp", "mean_temp", "max_humidex", "day_of_week", "day_of_year", "year"]
    
    print("🚀 Training Model A (Macro Daily Peak Forecaster)...")
    model_a = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.02, num_leaves=31, max_depth=6,
        random_state=42, n_jobs=-1, verbose=-1
    )
    model_a.fit(daily_train[macro_features], daily_train["actual_daily_peak"])
    
    # Generate true out-of-sample daily peak predictions
    daily_test["pred_daily_peak"] = model_a.predict(daily_test[macro_features])
    predicted_peaks_dict = dict(zip(daily_test["date"], daily_test["pred_daily_peak"]))
    
    # =================================================================
    # 🧠 PHASE 2: TRAIN MODEL B (MICRO HOURLY MODEL) & RECONCILE
    # =================================================================
    hourly_features = [col for col in df.columns if col not in ["timestamp", "coast_load_mw", "date"]]
    
    print("🚀 Training Model B (Micro Hourly Profile Forecaster)...")
    model_b = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.015, num_leaves=45, max_depth=7,
        min_child_samples=35, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.5, random_state=42, n_jobs=-1, verbose=-1
    )
    model_b.fit(train_df[hourly_features], train_df["coast_load_mw"])
    
    # Generate standard un-reconciled hourly predictions
    test_df["pred_raw"] = model_b.predict(test_df[hourly_features])
    
    # Group raw hourly predictions to find their un-reconciled peaks
    daily_pred_raw_peaks = test_df.groupby("date")["pred_raw"].max().to_dict()
    
    print("🛠️ Applying Production-Grade Peak-Scaling Reconciliation Layer...")
    reconciled_preds = []
    
    for idx, row in test_df.iterrows():
        current_date = row["date"]
        
        # OPERATIONAL GUARDRAIL: Use Model A's predicted peak as the constraint
        predicted_macro_peak = predicted_peaks_dict[current_date]
        predicted_micro_peak = daily_pred_raw_peaks[current_date]
        
        # Calculate scaling factor purely from predictions vs predictions
        production_scaling_factor = predicted_macro_peak / predicted_micro_peak
        
        reconciled_hourly_val = row["pred_raw"] * production_scaling_factor
        reconciled_preds.append(reconciled_hourly_val)
        
    test_df["pred_reconciled"] = reconciled_preds
    
    # =================================================================
    # 📊 PERFORMANCE CARD METRICS
    # =================================================================
    raw_rmse = np.sqrt(np.mean((test_df["coast_load_mw"] - test_df["pred_raw"]) ** 2))
    recon_rmse = np.sqrt(np.mean((test_df["coast_load_mw"] - test_df["pred_reconciled"]) ** 2))
    
    print("\n" + "="*55)
    # Highlight that this is a true live-simulation prediction out to Dec 2025
    print("🏆 PRODUCTION TEMPORAL HIERARCHY RECONCILIATION CARD")
    print("="*55)
    print(f"Base LightGBM Hourly RMSE:       {raw_rmse:.2f} Megawatts")
    print(f"Production Reconciled RMSE:      {recon_rmse:.2f} Megawatts")
    print(f"Realized Error Reduction:        {((raw_rmse - recon_rmse) / raw_rmse) * 100:.2f}%")
    print("="*55)
    
    print("\n🔍 Live Simulation Matrix Verification (First 5 Hours):")
    print(test_df[["timestamp", "coast_load_mw", "pred_raw", "pred_reconciled"]].head().to_string(index=False))

if __name__ == "__main__":
    run_production_reconciliation_pipeline()