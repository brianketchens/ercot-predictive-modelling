import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, r2_score
import lightgbm as lgb

def run_decade_validation_engine(data_path):
    """Executes a robust walk-forward cross-validation loop over the 10-year

    dataset, comparing a baseline Ridge model against an advanced LightGBM model.
    """
    print("Loading 10-Year Master Feature Matrix...")
    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    
    # 1. ISOLATE TARGET AND FEATURES
    # Our target variable is the actual Houston coast grid load in Megawatts
    y = df["coast_load_mw"]
    
    # Drop columns that are targets, or indexes that shouldn't be trained on
    X = df.drop(columns=["coast_load_mw"])
    
    print(f"Matrix shape: {X.shape[0]} rows across {X.shape[1]} advanced features.")
    
    # 2. INITIALIZE TIME-SERIES SPLITTER (Walk-Forward Validation)
    # n_splits=4 creates 5 sequential chronological windows over our decade
    tscv = TimeSeriesSplit(n_splits=4)
    
    # Tracking buckets for our cross-validation performance metrics
    baseline_rmses, baseline_r2s = [], []
    champion_rmses, champion_r2s = [], []
    
    print("\nStarting Walk-Forward Cross-Validation Loop...")
    print("-" * 65)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Chronological slicing prevents data leakage
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        train_start, train_end = X_train.index.min().strftime('%Y-%m'), X_train.index.max().strftime('%Y-%m')
        test_start, test_end = X_test.index.min().strftime('%Y-%m'), X_test.index.max().strftime('%Y-%m')
        
        print(f"Fold {fold+1}: Train [{train_start} to {train_end}] ➡️ Test [{test_start} to {test_end}]")
        
        # 3. TRAIN BASELINE (Ridge Regression)
        baseline_model = Ridge()
        baseline_model.fit(X_train, y_train)
        base_preds = baseline_model.predict(X_test)
        
        # 4. TRAIN CHAMPION (LightGBM Gradient Boosting)
        # We adjust hyperparameters slightly to account for the massive decade timeline
        champion_model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        champion_model.fit(X_train, y_train)
        champ_preds = champion_model.predict(X_test)
        
        # 5. SCORE THE FOLD
        fold_base_rmse = root_mean_squared_error(y_test, base_preds)
        fold_base_r2 = r2_score(y_test, base_preds)
        fold_champ_rmse = root_mean_squared_error(y_test, champ_preds)
        fold_champ_r2 = r2_score(y_test, champ_preds)
        
        baseline_rmses.append(fold_base_rmse)
        baseline_r2s.append(fold_base_r2)
        champion_rmses.append(fold_champ_rmse)
        champion_r2s.append(fold_champ_r2)
        
        print(f"  [Ridge]    RMSE: {fold_base_rmse:.2f} MW | R²: {fold_base_r2:.4f}")
        print(f"  [LightGBM] RMSE: {fold_champ_rmse:.2f} MW | R²: {fold_champ_r2:.4f}\n")
        
    print("-" * 65)
    print("🏆 FINAL AGGREGATED DECADE PERFORMANCE CARD 🏆")
    print("-" * 65)
    print(f"Baseline Ridge Regression Model:")
    print(f"  -> Mean RMSE: {np.mean(baseline_rmses):.2f} Megawatts")
    print(f"  -> Mean R² Score: {np.mean(baseline_r2s):.4f}")
    print(f"\nChampion LightGBM Machine Learning Model:")
    print(f"  -> Mean RMSE: {np.mean(champion_rmses):.2f} Megawatts")
    print(f"  -> Mean R² Score: {np.mean(champion_r2s):.4f}")
    print("-" * 65)
    
    # Calculate performance jump
    rmse_improvement = ((np.mean(baseline_rmses) - np.mean(champion_rmses)) / np.mean(baseline_rmses)) * 100
    print(f"Machine Learning slashed error by {rmse_improvement:.2f}% compared to baseline!")

if __name__ == "__main__":
    processed_matrix = os.path.join("data", "processed", "engineered_energy_features.csv")
    run_decade_validation_engine(data_path=processed_matrix)