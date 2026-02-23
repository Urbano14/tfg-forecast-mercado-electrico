import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/esios_600_spot_diario_ES.parquet")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQUET = OUT_DIR / "spot_es_processed.parquet"
OUT_CSV = OUT_DIR / "spot_es_processed.csv"

df = pd.read_parquet(RAW_PATH)

# 1) timestamp en UTC
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp_utc"])

# 2) convertir a hora local España 
df["timestamp"] = df["timestamp_utc"].dt.tz_convert("Europe/Madrid")

# 3) quedarnos con lo mínimo y ordenar
df = df[["timestamp", "price"]].sort_values("timestamp").reset_index(drop=True)

# 4) diagnóstico
expected = int((df["timestamp"].max() - df["timestamp"].min()) / pd.Timedelta(hours=1)) + 1
print("Horas esperadas (rango):", expected)
print("Horas reales (filas):", len(df))

dups = df["timestamp"].duplicated().sum()
print("Duplicados:", dups)

# prevenir problemas por duplicados
df = df.drop_duplicates(subset=["timestamp"], keep="last")

df2 = df.set_index("timestamp").asfreq("h")
missing = df2["price"].isna().sum()
print("Missing horas tras asfreq(h):", missing)

# Guardar processed
df2.reset_index().to_parquet(OUT_PARQUET, index=False)
df2.reset_index().to_csv(OUT_CSV, index=False)

print("Processed guardado")
print("Desde:", df2.index.min(), "| Hasta:", df2.index.max(), "| Filas:", len(df2))
print(df2.head())