from pathlib import Path

import pandas as pd

# Este script fusiona el dataset de precios con el dataset exógeno,
# usando la columna "timestamp" como clave de unión.
# El resultado se guarda en formato parquet y csv.

PRICE_PATH = Path("data/processed/spot_es_processed.parquet")
EXOG_PATH = Path("data/raw/exogenous/exogenous_merged.parquet")

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQUET = OUT_DIR / "spot_es_with_exogenous.parquet"
OUT_CSV = OUT_DIR / "spot_es_with_exogenous.csv"
TIMEZONE = "Europe/Madrid"
FINAL_COLUMNS = [
    "timestamp",
    "price",
    "demand_forecast",
    "wind_forecast",
    "solar_forecast",
    "hydro_programmed",
]


def ensure_europe_madrid(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize(TIMEZONE)
    return timestamps.dt.tz_convert(TIMEZONE)


# Carga el dataset de precios.
def load_price() -> pd.DataFrame:
    df = pd.read_parquet(PRICE_PATH)
    df["timestamp"] = ensure_europe_madrid(df["timestamp"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price"])
    df = df[["timestamp", "price"]]
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# Carga el dataset exógeno.
def load_exogenous() -> pd.DataFrame:
    df = pd.read_parquet(EXOG_PATH)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df["timestamp"] = df["timestamp_utc"].dt.tz_convert(TIMEZONE)
    df = df.drop(columns=["timestamp_utc"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main() -> None:
    price = load_price()
    exog = load_exogenous()

    df = price.merge(exog, on="timestamp", how="left")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[FINAL_COLUMNS]

    print("Shape tras merge:", df.shape)
    print("Timestamp mínimo:", df["timestamp"].min())
    print("Timestamp máximo:", df["timestamp"].max())
    print("Duplicados:", int(df["timestamp"].duplicated().sum()))
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
