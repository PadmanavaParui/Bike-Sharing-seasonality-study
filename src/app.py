import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(page_title="Bike Demand Dashboard", layout="wide")

# Define Data Paths
CLEAN_DATA_PATH = "data/gold_standard_bike_data.csv"
FORECAST_PATH = "data/output/prophet_forecast_tuned.csv"

@st.cache_data
def load_data():
    """Loads and prepares the data. Cached so it doesn't reload on every button click."""
    if not os.path.exists(CLEAN_DATA_PATH):
        st.error("Error: Clean data not found. Please run the pipeline first.")
        st.stop()
        
    df = pd.read_csv(CLEAN_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Aggregate to daily for the main views
    daily_df = df.groupby(df['datetime'].dt.date).agg({
        'cnt': 'sum',
        'registered': 'sum',
        'casual': 'sum',
        'temp': 'mean'
    }).reset_index()
    daily_df['datetime'] = pd.to_datetime(daily_df['datetime'])
    
    return df, daily_df

@st.cache_data
def load_forecast():
    """Loads the Prophet forecast data."""
    if os.path.exists(FORECAST_PATH):
        fc = pd.read_csv(FORECAST_PATH)
        fc['ds'] = pd.to_datetime(fc['ds'])
        return fc
    return None

# Load Data
raw_df, daily_df = load_data()
forecast_df = load_forecast()

# --- HEADER ---
st.title("🚲 Bike Sharing Seasonality & Demand Dashboard")
st.markdown("Analyze historical demand, seasonal trends, and 30-day forecasts.")
st.divider()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Dashboard Controls")
date_range = st.sidebar.date_input(
    "Select Date Range (Historical Data)",
    value=(daily_df['datetime'].min(), daily_df['datetime'].max()),
    min_value=daily_df['datetime'].min(),
    max_value=daily_df['datetime'].max()
)

# Filter the dataframe based on sidebar dates
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (daily_df['datetime'].dt.date >= start_date) & (daily_df['datetime'].dt.date <= end_date)
    filtered_daily = daily_df.loc[mask]
else:
    filtered_daily = daily_df

# --- TOP METRICS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rides in Period", f"{filtered_daily['cnt'].sum():,}")
with col2:
    st.metric("Avg Daily Registered Rides", f"{int(filtered_daily['registered'].mean()):,}")
with col3:
    st.metric("Avg Daily Casual Rides", f"{int(filtered_daily['casual'].mean()):,}")

# --- MAIN VISUALIZATIONS ---
st.subheader("Demand Segmentation: Registered vs. Casual")
# PRD Requirement: Segment drill-downs
fig_segment = px.area(
    filtered_daily, 
    x="datetime", 
    y=["registered", "casual"],
    labels={"value": "Number of Rides", "variable": "Rider Type", "datetime": "Date"},
    color_discrete_map={"registered": "#1f77b4", "casual": "#ff7f0e"}
)
st.plotly_chart(fig_segment, use_container_width=True)


st.subheader("30-Day Demand Forecast (Prophet Model)")
if forecast_df is not None:
    # PRD Requirement: Uncertainty Quantification visually displayed
    fig_forecast = go.Figure()
    
    # Add historical actuals
    fig_forecast.add_trace(go.Scatter(
        x=daily_df['datetime'], y=daily_df['cnt'],
        mode='lines', name='Historical Actuals', line=dict(color='black')
    ))
    
    # Add Forecast line
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['ds'], y=forecast_df['yhat'],
        mode='lines', name='Forecasted Demand', line=dict(color='blue')
    ))
    
    # Add Confidence Interval shading
    fig_forecast.add_trace(go.Scatter(
        x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
        y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Uncertainty Interval'
    ))
    
    fig_forecast.update_layout(xaxis_title="Date", yaxis_title="Daily Rides")
    # Zoom in to the last few months of actuals + the forecast window
    fig_forecast.update_xaxes(range=[daily_df['datetime'].max() - pd.Timedelta(days=180), forecast_df['ds'].max()])
    
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.warning("Forecast data not found. Please run the forecasting script first.")