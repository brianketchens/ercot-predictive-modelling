import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def generate_mock_energy_data():
    """Generates a synthetic hourly dataset to demonstrate walk-forward splitting.

    Mimics a cleaned dataframe containing ERCOT load and Open-Meteo features.
    """
    print("Generating mock time-series data...")
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="h")
    np.random.seed(42)

    df = pd.DataFrame(index=dates)
    df["temperature"] = 70 + 15 * np.sin(df.index.dayofyear / 365 * 2 * np.pi) + np.random.normal(0, 5, len(dates))
    df["hour_of_day"] = df.index.hour
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
    df["load_lag_24h"] = 30000 + 500 * df["temperature"] + np.random.normal(0, 1000, len(dates))
    
    # Target variable: ERCOT actual hourly load
    df["ercot_load"] = 30500 + 520 * df["temperature"] + 200 * df["hour_of_day"] + np.random.normal(0, 800, len(dates))
    
    return df


def run_walk_forward_validation(df, target_col, feature_cols, n_splits=5):
    """Executes an expanding window walk-forward validation loop.
    
    Args:
        df (pd.DataFrame): Sorted time-series DataFrame with a DatetimeIndex.
        target_col (str): The column name of the target variable (e.g., 'ercot_load').
        feature_cols (list): List of feature column names used for training.
        n_splits (int): Number of historical walk-forward chunks to evaluate.
    """
    # 1. Ensure chronological order (CRITICAL for time-series splits)
    df = df.sort_index()
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 2. Initialize the Scikit-Learn TimeSeriesSplitter
    # max_train_size can be set if you want a "rolling" window instead of "expanding"
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    metrics_summary = []
    
    print(f"\nStarting Walk-Forward Validation ({n_splits} chronological splits)")
    print("-" * 70)
    
    # 3. The Iterative Loop
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Splitting data via positional indices
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Get date ranges for console logging transparency
        train_start, train_end = X_train.index.min(), X_train.index.max()
        test_start, test_end = X_test.index.min(), X_test.index.max()
        
        print(f"Fold {fold + 1}:")
        print(f"  Train Window: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} ({len(X_train)} hours)")
        print(f"  Test Window:  {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} ({len(X_test)} hours)")
        
        # 4. Initialize and Train the Model (Using Ridge Regression as a strong baseline)
        model = model_object
        model.fit(X_train, y_train)
        
        # 5. Generate Predictions
        predictions = model.predict(X_test)
        
        # 6. Evaluate Performance for this iteration
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"  Evaluation Metrics -> RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW | R²: {r2:.4f}\n")
        
        # Store metadata and metrics
        metrics_summary.append({
            "fold": fold + 1,
            "train_hours": len(X_train),
            "test_hours": len(X_test),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "model_object": model  # Useful if you want to inspect coefficients/trees later
        })
        
    # 7. Aggregate and Summarize Overall Performance
    metrics_df = pd.DataFrame(metrics_summary)
    print("-" * 70)
    print("OVERALL PIPELINE METRICS (AVERAGE ACROSS ALL WALKS):")
    print(f"  Mean RMSE: {metrics_df['rmse'].mean():.2f} MW")
    print(f"  Mean MAE:  {metrics_df['mae'].mean():.2f} MW")
    print(f"  Mean R²:   {metrics_df['r2'].mean():.4f}")
    print("-" * 70)
    
    return metrics_df


if __name__ == "__main__":
    # Simulate data ingestion step
    data = generate_mock_energy_data()
    
    # Define our analytical columns
    target = "ercot_load"
    features = ["temperature", "hour_of_day", "is_weekend", "load_lag_24h"]
    
    # Run pipeline validation
    results = run_walk_forward_validation(data, target_col=target, feature_cols=features, n_splits=5)