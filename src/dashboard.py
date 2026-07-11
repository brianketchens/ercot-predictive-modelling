import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error

# Set wide layout for an executive control-room feel
st.set_page_config(layout="wide", page_title="BriAnalyst Grid Insights Engine")

@st.cache_data
def load_and_predict_data():
    processed_matrix = os.path.join("data", "processed", "master_features_expanded.csv")
    if not os.path.exists(processed_matrix):
        st.error(f"Missing master features matrix at {processed_matrix}. Please run features pipeline first.")
        st.stop()

    df = pd.read_csv(processed_matrix, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 🧼 CLEANING LAYER: Drop any row with missing target loads or features
    df = df.dropna().reset_index(drop=True)

    df["date"] = df["timestamp"].dt.date

    # Chronological Split (Fold 4 Window)
    split_date = pd.to_datetime("2024-02-01")
    train_df = df[df["timestamp"] < split_date].reset_index(drop=True)
    test_df = df[df["timestamp"] >= split_date].reset_index(drop=True)

    features = [col for col in df.columns if col not in ["timestamp", "coast_load_mw", "date"]]

    # Fit Baseline Ridge
    ridge = Ridge()
    ridge.fit(train_df[features], train_df["coast_load_mw"])
    test_df["pred_ridge"] = ridge.predict(test_df[features])

    # Fit Champion LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.015, num_leaves=45,
        max_depth=7, min_child_samples=35, subsample=0.7,
        colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.5,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(train_df[features], train_df["coast_load_mw"])
    test_df["pred_lgb"] = lgb_model.predict(test_df[features])

    # Calculate Feature Importance Metrics
    feat_imp_df = pd.DataFrame({
        'feature': features,
        'importance_gain': lgb_model.booster_.feature_importance(importance_type='gain')
    })
    feat_imp_df['pct'] = (feat_imp_df['importance_gain'] / feat_imp_df['importance_gain'].sum()) * 100
    feat_imp_df = feat_imp_df.sort_values('pct', ascending=True)

    # === LIVE PERFORMANCE METRICS ===
    # Computed directly from the hold-out predictions so the KPI ribbon can
    # never drift away from what the model actually produces.
    ridge_rmse = root_mean_squared_error(test_df["coast_load_mw"], test_df["pred_ridge"])
    lgb_rmse = root_mean_squared_error(test_df["coast_load_mw"], test_df["pred_lgb"])
    lgb_r2 = r2_score(test_df["coast_load_mw"], test_df["pred_lgb"])

    metrics = {
        "lgb_r2": lgb_r2,
        "rmse_reduction_pct": (ridge_rmse - lgb_rmse) / ridge_rmse * 100,
        "rmse_delta": lgb_rmse - ridge_rmse,   # negative = error reduced vs baseline
        "n_features": len(features),
    }

    return test_df, feat_imp_df, metrics

# Load Data Pipeline
test_df, feat_imp_df, metrics = load_and_predict_data()

# =================================================================
# 🎨 STREAMLIT DASHBOARD UI RENDERING
# =================================================================
st.title("⚡ ERCOT Houston Coast Zone Load Insights Engine")
st.markdown("##### *Continuous Improvement & Grid Operations Consulting Baseline*")
st.write("---")

# 1. Executive KPI Ribbon (all values computed live from the hold-out set)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Champion Model R² (Hold-Out)",
        value=f"{metrics['lgb_r2']:.4f}",
        delta="🏆 LightGBM"
    )
with col2:
    st.metric(
        label="RMSE Reduction vs. Ridge",
        value=f"{metrics['rmse_reduction_pct']:.2f}%",
        delta=f"{metrics['rmse_delta']:.1f} MW",
        delta_color="inverse"  # lower RMSE is better, so show the drop as green
    )
with col3:
    st.metric(label="Out-of-Sample Window", value="Feb 2024 – Dec 2025")
with col4:
    st.metric(label="Engineered Features", value=f"{metrics['n_features']} Dimensions")

st.write("---")

# 2. Main Sidebar Navigation & Date Filters
st.sidebar.header("🕹️ Operations Control Center")
selected_date = st.sidebar.date_input(
    "Select Target Forecast Date",
    value=pd.to_datetime("2024-07-15").date(),
    min_value=test_df["date"].min(),
    max_value=test_df["date"].max()
)

# Filter dataset down to the single selected operational day
day_mask = test_df["date"] == selected_date
day_data = test_df[day_mask].sort_values("timestamp")

if day_data.empty:
    st.warning("⚠️ No data available for the selected timeline boundary.")
else:
    # 3. Interactive Plotly Forecast Canvas
    st.subheader(f"📊 24-Hour Horizon Analysis: {selected_date}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=day_data["timestamp"], y=day_data["coast_load_mw"], name="Actual Grid Load", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=day_data["timestamp"], y=day_data["pred_lgb"], name="Champion LightGBM", line=dict(color="#2ca02c", width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(x=day_data["timestamp"], y=day_data["pred_ridge"], name="Baseline Ridge Regression", line=dict(color="#d62728", width=1.5, dash="dot")))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Operational Hour",
        yaxis_title="Grid Load (Megawatts)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=450
    )
    st.plotly_chart(fig, width='stretch')

st.write("---")

# 4. Thermodynamic Feature Analysis & Operational Post-Mortem Rows
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("🔥 Meteorological Information Gain Card")
    fig_imp = go.Figure(go.Bar(
        x=feat_imp_df['pct'],
        y=feat_imp_df['feature'],
        orientation='h',
        marker_color='#ff7f0e'
    ))
    fig_imp.update_layout(
        template="plotly_dark",
        xaxis_title="Relative Contribution Percentage (%)",
        margin=dict(l=40, r=40, t=10, b=40),
        height=380
    )
    st.plotly_chart(fig_imp, width='stretch')

with right_col:
    st.subheader("💡 What the Model Is Keying On")
    st.info(
        "**Reading the forecast drivers:**\n\n"
        "1. **Temperature leads, engineered comfort features back it up.** Air temperature is still the "
        "single strongest signal (~43% of the model's decision gain) — exactly what you'd expect for a "
        "summer-peaking, AC-heavy grid. Layered on top, the engineered comfort features carry real weight: "
        "cooling degree hours (~21%) and humidex (~12%) capture the non-linear jump in demand once "
        "temperatures push past the comfort threshold.\n\n"
        "2. **Regularization bought stability, not just accuracy.** Capping tree depth (`max_depth=7`) and "
        "requiring larger leaves (`min_child_samples=35`) keeps the model from chasing one-off anomalies, "
        "so it generalizes across the full decade instead of memorizing individual heat events.\n\n"
        "3. **Why we forecast hourly load directly.** A two-stage design — predict the daily peak, then "
        "rescale the hourly shape to match — sounds appealing, but in testing a miss on the predicted peak "
        "propagated into every hour of that day. Forecasting hourly load directly stayed more robust, so "
        "that's the configuration we ship."
    )

    # Export the validation matrix straight to the user's browser.
    # (download_button streams the file to the client, so it works on
    # Streamlit Community Cloud's ephemeral filesystem.)
    st.subheader("📥 Export Validation Matrix")
    csv_bytes = test_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Validation Predictions (CSV)",
        data=csv_bytes,
        file_name="ercot_validation_predictions.csv",
        mime="text/csv"
    )
