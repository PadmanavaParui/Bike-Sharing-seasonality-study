import pandas as pd
import requests
import zipfile
import io
import os


DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
RAW_DATA_PATH = "data/hour.csv"
CLEAN_DATA_PATH = "data/gold_standard_bike_data.csv"

"""
    downloading the UCI Bike Sharing dataset and extracting it to the data folder.
"""
def download_and_extract_data():
    print("Downloading dataset from UCI...")
    response = requests.get(DATA_URL)


    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall("data/")
        print("Download and extraction complete.")
    else:
        raise Exception(f"Failed to download data. Status code: {response.status_code}")
    
"""
    loading the hourly data, and enforcing the schema, and interpolates missing timestamps.
"""
def load_and_clean_data():
    print("Loading raw data...")

    df = pd.read_csv(RAW_DATA_PATH)


    df['dteday'] = pd.to_datetime(df['dteday'])


    df['datetime'] = df['dteday']+pd.to_timedelta(df['hr'], unit = 'h')
    df.set_index('datetime', inplace = True)
    df.drop(['dteday', 'hr'], axis = 1, inplace = True)

    
    print("Checking for missing timestamps and interpolating...")
    full_time_range = pd.date_range(start = df.index.min(), end = df.index.max(), freq = "h")

    df_reindexed = df.reindex(full_time_range)

    df_clean = df_reindexed.interpolate(method = 'linear')



    print(f"Saving cleaned data to {CLEAN_DATA_PATH}...")

    df_clean.to_csv(CLEAN_DATA_PATH, index_label = 'datetime')
    print("Ingestion and cleaning complete! Gold Standard dataset is ready.")

if __name__ == "__main__":
    # ensuring the data directory exists
    os.makedirs("data", exist_ok=True)

    #Running the pipeline steps
    download_and_extract_data()
    load_and_clean_data()