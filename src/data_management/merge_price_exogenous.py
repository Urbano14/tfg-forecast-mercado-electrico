from pathlib import Path
import pandas as pd


PRICE_PATH = Path("data/processed/spot_es_processed.parquet")
EXOG_PATH = Path("data/raw/exogenous/exogenous_merged.parquet")

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQUET = OUT_DIR / "spot_es_with_exogenous.parquet"
OUT_CSV = OUT_DIR / "spot_es_with_exogenous.csv"


def load_price() -> pd.DataFrame:
    df = pd.read_parquet(PRICE_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Madrid")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_exogenous() -> pd.DataFrame:
    df = pd.read_parquet(EXOG_PATH)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp"] = df["timestamp_utc"].dt.tz_convert("Europe/Madrid")

    df = df.drop(columns=["timestamp_utc"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    price = load_price()
    exog = load_exogenous()

    df = price.merge(exog, on="timestamp", how="left")

    print("Shape tras merge:", df.shape)
    print("\nMissing por columna:")
    print(df.isna().sum())

    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    print("\nOK. Dataset fusionado guardado en:")
    print("-", OUT_PARQUET)
    print("-", OUT_CSV)
    print("\nPrimeras filas:")
    print(df.head())


if __name__ == "__main__":
    main()