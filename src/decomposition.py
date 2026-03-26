import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import os

# Define paths
CLEAN_DATA_PATH = "data/gold_standard_bike_data.csv"
OUTPUT_DIR = "data/output"

def check_stationarity(timeseries):
    """Runs the Augmented Dickey-Fuller test to check for unit roots."""
    print("\n--- Stationarity Check (ADF Test) ---")
    
    # adfuller expects a 1D array/series without NaNs
    result = adfuller(timeseries.dropna())
    
    p_value = result[1]
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # Checking against the 0.05 threshold from your PRD
    if p_value <= 0.05:
        print("Conclusion: The time series is STATIONARY (p-value <= 0.05).")
        print("We do NOT need to apply differencing before modeling.")
    else:
        print("Conclusion: The time series is NON-STATIONARY (p-value > 0.05).")
        print("We MUST apply differencing before modeling to avoid spurious results.")

def run_decomposition():
    print(f"Loading clean data from {CLEAN_DATA_PATH}...")
    # Load data and set datetime as the index so statsmodels can understand the timeline
    df = pd.read_csv(CLEAN_DATA_PATH, index_col='datetime', parse_dates=True)
    
    # PRD Requirement: Decompose daily rentals.
    print("Aggregating hourly data to daily frequency...")
    # We only sum the 'cnt' column to avoid miscalculating weather averages
    daily_series = df[['cnt']].resample('D').sum()['cnt'] 
    
    # 1. Run the Stationarity Check
    check_stationarity(daily_series)
    
    # 2. Run STL Decomposition
    print("\n--- Running STL Decomposition ---")
    # We set period=7 to extract the weekly seasonality (since the data is now daily)
    stl = STL(daily_series, period=7, robust=True) 
    result = stl.fit()
    
    print("Decomposition complete. Saving plots and data...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save the visual plot
    fig = result.plot()
    fig.set_size_inches(10, 8)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "stl_decomposition_plot.png")
    plt.savefig(plot_path)
    print(f"Visual Plot saved to: {plot_path}")
    
    # Save the mathematical components to a new CSV for the modeling phase
    components_df = pd.DataFrame({
        'Actual': daily_series,
        'Trend': result.trend,
        'Seasonal': result.seasonal,
        'Residual': result.resid
    })
    
    components_path = os.path.join(OUTPUT_DIR, "decomposed_components.csv")
    components_df.to_csv(components_path)
    print(f"Mathematical Components saved to: {components_path}")

if __name__ == "__main__":
    run_decomposition()