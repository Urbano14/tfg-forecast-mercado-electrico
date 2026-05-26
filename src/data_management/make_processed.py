import pandas as pd
from pathlib import Path

# Este script toma el dataset de precios descargado en formato parquet, 
# lo procesa para asegurarse de que tiene un timestamp horario continuo sin duplicados ni missing,
#  y lo guarda en formato parquet y csv para su uso posterior.

RAW_PATH = Path("data/raw/esios_600_spot_diario_ES.parquet")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQUET = OUT_DIR / "spot_es_processed.parquet"
OUT_CSV = OUT_DIR / "spot_es_processed.csv"


df = pd.read_parquet(RAW_PATH)

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp_utc"])

df["timestamp"] = df["timestamp_utc"].dt.tz_convert("Europe/Madrid")

df = df[["timestamp", "price"]].sort_values("timestamp").reset_index(drop=True)

expected = int((df["timestamp"].max() - df["timestamp"].min()) / pd.Timedelta(hours=1)) + 1
print("Horas esperadas (rango):", expected)
print("Horas reales (filas):", len(df))

dups = df["timestamp"].duplicated().sum()
print("Duplicados:", dups)

df = df.drop_duplicates(subset=["timestamp"], keep="last")

df2 = df.set_index("timestamp").asfreq("h")
missing = df2["price"].isna().sum()
print("Missing horas tras asfreq(h):", missing)

df2.reset_index().to_parquet(OUT_PARQUET, index=False)
df2.reset_index().to_csv(OUT_CSV, index=False)

print("Processed guardado")
print("Desde:", df2.index.min(), "| Hasta:", df2.index.max(), "| Filas:", len(df2))
print(df2.head())