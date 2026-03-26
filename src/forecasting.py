import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import os
import matplotlib.pyplot as plt

# Define paths
CLEAN_DATA_PATH = "data/gold_standard_bike_data.csv"
OUTPUT_DIR = "data/output"

def run_prophet_model():
    print(f"Loading data from {CLEAN_DATA_PATH}...")
    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=['datetime'])
    
    # We need to aggregate the target AND the exogenous variables to daily
    # For weather, we take the mean. For counts and holidays, we take the max/sum.
    daily_df = df.groupby(df['datetime'].dt.date).agg({
        'cnt': 'sum',
        'temp': 'mean',
        'hum': 'mean',
        'windspeed': 'mean',
        'holiday': 'max' # If any hour is a holiday, the whole day is a holiday
    }).reset_index()
    
    # Rename for Prophet
    prophet_df = daily_df.rename(columns={'datetime': 'ds', 'cnt': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    print("\n--- Training Tuned Prophet Model ---")
    # Initialize the model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    
    # ADD EXOGENOUS REGRESSORS
    # This tells Prophet: "Hey, look at the weather and holidays before you guess the demand!"
    model.add_regressor('temp')
    model.add_regressor('hum')
    model.add_regressor('windspeed')
    model.add_regressor('holiday')
    
    # Fit the model
    model.fit(prophet_df)
    
    print("\n--- Running Sliding Window Validation (Cross-Validation) ---")
    cv_results = cross_validation(
        model, 
        initial='365 days',
        period='30 days',
        horizon='30 days'
    )
    
    df_p = performance_metrics(cv_results)
    prophet_mape = df_p['mape'].mean() * 100 
    
    print(f"\n🏆 Tuned Prophet Cross-Validated MAPE: {prophet_mape:.2f}%")
    
    if prophet_mape < 22.86:
        print("✅ SUCCESS! The tuned model successfully beat the Naive Baseline (22.86%).")
    else:
        print("❌ The model still needs tuning. It did not beat the baseline.")
        
    print("\n--- Generating Future Forecast with Uncertainty ---")
    # Make future dataframe
    future = model.make_future_dataframe(periods=30)
    
    # CRITICAL: We have to provide future "guesses" for our exogenous variables
    # For a real system, you'd plug in a weather forecast here. 
    # For this project, we'll just carry forward the last known weather values.
    last_known_temp = prophet_df['temp'].iloc[-1]
    last_known_hum = prophet_df['hum'].iloc[-1]
    last_known_wind = prophet_df['windspeed'].iloc[-1]
    
    # Fill in the future regressors
    future['temp'] = prophet_df['temp']
    future['temp'].fillna(last_known_temp, inplace=True)
    
    future['hum'] = prophet_df['hum']
    future['hum'].fillna(last_known_hum, inplace=True)
    
    future['windspeed'] = prophet_df['windspeed']
    future['windspeed'].fillna(last_known_wind, inplace=True)
    
    future['holiday'] = prophet_df['holiday']
    future['holiday'].fillna(0, inplace=True) # Assume no holidays in the next 30 days
    
    # Predict
    forecast = model.predict(future)
    
    print(f"Saving forecast data and plots to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    forecast.to_csv(os.path.join(OUTPUT_DIR, "prophet_forecast_tuned.csv"), index=False)
    
    # Save the plot
    fig1 = model.plot(forecast)
    plt.title("Tuned Bike Demand Forecast (Next 30 Days)")
    plt.savefig(os.path.join(OUTPUT_DIR, "prophet_forecast_tuned_plot.png"))
    
    latest_pred = forecast.iloc[-1]
    print(f"\nUncertainty Quantification for {latest_pred['ds'].date()}:")
    print(f"Predicted Demand: {latest_pred['yhat']:.0f} bikes")
    print(f"Confidence Range: {latest_pred['yhat_lower']:.0f} to {latest_pred['yhat_upper']:.0f} bikes")

if __name__ == "__main__":
    run_prophet_model()