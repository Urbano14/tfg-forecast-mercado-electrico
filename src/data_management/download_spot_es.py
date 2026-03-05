from __future__ import annotations

from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import datetime, timedelta
import requests
import pandas as pd

# --- Config ---
INDICATOR_ID = 600
GEO_ID_ES = 3

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True) 

OUTPUT_PARQUET = RAW_DIR / f"esios_{INDICATOR_ID}_spot_diario_ES.parquet"
OUTPUT_CSV = RAW_DIR / f"esios_{INDICATOR_ID}_spot_diario_ES.csv"


CHUNK_DAYS = 30

DEFAULT_START = "2020-01-01T00:00"
DEFAULT_END = "2024-12-31T23:59"


def esios_get(token: str, params: dict) -> dict: 
    url = f"https://api.esios.ree.es/indicators/{INDICATOR_ID}"
    headers = {
        "Accept": "application/json; application/vnd.esios-api-v2+json",
        "Content-Type": "application/json",
        "x-api-key": token,
        "Authorization": f"Token token={token}",
    }
    r = requests.get(url, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def values_to_df(payload: dict) -> pd.DataFrame:
    values = payload.get("indicator", {}).get("values", [])
    if not values:
        return pd.DataFrame(columns=["timestamp_utc", "price", "geo_id", "geo_name"])

    df = pd.json_normalize(values)
    df = df.rename(columns={"value": "price", "datetime_utc": "timestamp_utc"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df[["timestamp_utc", "price", "geo_id", "geo_name"]].dropna()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def parse_iso_local(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M")


def to_iso_local(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M")


def main():
    load_dotenv()
    token = os.getenv("ESIOS_TOKEN")
    if not token:
        raise SystemExit("Falta ESIOS_TOKEN en .env")

    if OUTPUT_PARQUET.exists():
        print(f"Ya existe {OUTPUT_PARQUET}. No descargo de nuevo para evitar peticiones redundantes.")
        print("Si quieres re-descargar, borra ese fichero primero.")
        return

    start = os.getenv("ESIOS_START", DEFAULT_START)
    end = os.getenv("ESIOS_END", DEFAULT_END)

    start_dt = parse_iso_local(start)
    end_dt = parse_iso_local(end) if end else datetime.now()

    print(f"Descargando indicador {INDICATOR_ID} (España geo_id={GEO_ID_ES})")
    print(f"Rango: {start_dt} -> {end_dt} (chunks de {CHUNK_DAYS} días)")

    all_chunks = []
    cur = start_dt

    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS), end_dt)

        params = {
            "start_date": to_iso_local(cur),
            "end_date": to_iso_local(chunk_end),
            "time_trunc": "hour",
            "geo_ids[]": GEO_ID_ES,
        }

        payload = esios_get(token, params)
        df_chunk = values_to_df(payload)

        print(f"- {to_iso_local(cur)} -> {to_iso_local(chunk_end)} | filas: {len(df_chunk)}")

        all_chunks.append(df_chunk)
        cur = chunk_end

    df = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()

    df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)

    # Guardar
    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nOK ✅ Guardado raw:")
    print(f"- {OUTPUT_PARQUET}")
    print(f"- {OUTPUT_CSV}")
    if not df.empty:
        print(f"Filas: {len(df)} | Desde: {df['timestamp_utc'].min()} | Hasta: {df['timestamp_utc'].max()}")
        print(df.head())


if __name__ == "__main__":
    main()
    