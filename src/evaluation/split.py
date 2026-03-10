import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/spot_es_processed.parquet")

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def temporal_split(df: pd.DataFrame):
    train = df[(df["timestamp"].dt.year >= 2020) & (df["timestamp"].dt.year <= 2022)]
    val = df[df["timestamp"].dt.year == 2023]
    test = df[df["timestamp"].dt.year == 2024]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

