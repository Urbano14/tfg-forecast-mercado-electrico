import pandas as pd

df = pd.read_parquet("data/processed/spot_es_with_exogenous.parquet")

mask = (
    (df["timestamp"] >= "2024-12-30")
    & (df["timestamp"] <= "2025-01-03")
)

print(df.loc[mask, ["timestamp", "price"]].to_string(index=False))