import pandas as pd
import numpy as np
import os

# Define path to the decomposed data we generated in the last step
COMPONENTS_PATH = "data/output/decomposed_components.csv"

def calculate_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Translates error into business terms (e.g., "We are off by X% on average").
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Protect against division by zero just in case there are days with 0 rentals
    non_zero_mask = y_true != 0
    y_true_safe = y_true[non_zero_mask]
    y_pred_safe = y_pred[non_zero_mask]
    
    # Formula from PRD: (100% / n) * Sum(|(Actual - Forecast) / Actual|)
    mape = np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100
    return mape

def run_baselines():
    print(f"Loading actuals from {COMPONENTS_PATH}...")
    try:
        df = pd.read_csv(COMPONENTS_PATH, index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Error: Decomposed components not found. Please run decomposition.py first.")
        return
        
    actuals = df['Actual']
    
    print("Calculating Baseline Forecasts...")
    
    # 1. Naive Forecast: Tomorrow = Today
    # We simply shift the actuals forward by 1 day
    df['Naive_Forecast'] = actuals.shift(1)
    
    # 2. Moving Average Forecast: 7-day window
    # We take the rolling mean of the PREVIOUS 7 days to predict today
    df['MA7_Forecast'] = actuals.shift(1).rolling(window=7).mean()
    
    # Drop the first 7 days where we have NaN values due to the rolling window
    eval_df = df.dropna()
    
    print("\n--- Baseline Model Evaluation (Scores to Beat) ---")
    
    # Score the Naive Forecast
    naive_mape = calculate_mape(eval_df['Actual'], eval_df['Naive_Forecast'])
    print(f"Naive Forecast (Tomorrow = Today) MAPE:  {naive_mape:.2f}%")
    
    # Score the 7-Day Moving Average
    ma7_mape = calculate_mape(eval_df['Actual'], eval_df['MA7_Forecast'])
    print(f"7-Day Moving Average Forecast MAPE:      {ma7_mape:.2f}%")
    
    print("\nTake note of these numbers! When we build our advanced Prophet model,")
    print("it MUST achieve a lower MAPE than these to be considered a success.")

if __name__ == "__main__":
    run_baselines()