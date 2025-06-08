import pandas as pd

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

def load_raw_data():
    df = pd.read_parquet(DATA_URL)
    print(f"Loaded raw data shape: {df.shape}")
    return df
