from pathlib import Path

import pandas as pd

# Este script construye la serie final de precios combinando dos fuentes:
# ESIOS para 2020-2024, ya validado en el histórico,
# y OMIE desde 2025-01-01 hasta 2026-05-01, porque ESIOS presentaba
# valores no homogéneos desde 2025.

ESIOS_RAW_PATH = Path("data/raw/esios_600_spot_diario_ES.parquet")
OMIE_RAW_PATH = Path("data/raw/omie_spot_es.parquet")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQUET = OUT_DIR / "spot_es_processed.parquet"
OUT_CSV = OUT_DIR / "spot_es_processed.csv"
TIMEZONE = "Europe/Madrid"
OMIE_START = pd.Timestamp("2025-01-01 00:00:00", tz=TIMEZONE)


def ensure_europe_madrid(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize(TIMEZONE)
    return timestamps.dt.tz_convert(TIMEZONE)


def load_esios_price() -> pd.DataFrame:
    df = pd.read_parquet(ESIOS_RAW_PATH)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["timestamp"] = df["timestamp_utc"].dt.tz_convert(TIMEZONE)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[["timestamp", "price"]]
    df = df[df["timestamp"] < OMIE_START]
    return df


def load_omie_price() -> pd.DataFrame:
    df = pd.read_parquet(OMIE_RAW_PATH)
    df["timestamp"] = ensure_europe_madrid(df["timestamp"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[["timestamp", "price"]]
    df = df[df["timestamp"] >= OMIE_START]
    return df


def print_price_checks(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Timestamp mínimo:", df["timestamp"].min())
    print("Timestamp máximo:", df["timestamp"].max())
    print("Duplicados:", int(df["timestamp"].duplicated().sum()))
    print("Nulos en price:", int(df["price"].isna().sum()))

    yearly_stats = df.groupby(df["timestamp"].dt.year)["price"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    print("\nResumen anual de price:")
    print(yearly_stats)


def main() -> None:
    esios_df = load_esios_price()
    omie_df = load_omie_price()
    df = pd.concat([esios_df, omie_df], ignore_index=True)
    df = df.dropna(subset=["timestamp", "price"])
    df = df[["timestamp", "price"]]
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    print("Processed guardado")
    print("-", OUT_PARQUET)
    print("-", OUT_CSV)
    print("\nFuente ESIOS usada hasta 2024-12-31 23:00 Europe/Madrid.")
    print("Fuente OMIE usada desde 2025-01-01 00:00 Europe/Madrid.")
    print_price_checks(df)
    print("\nPrimeras filas:")
    print(df.head())


if __name__ == "__main__":
    main()
