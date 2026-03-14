from __future__ import annotations

from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import datetime, timedelta
import requests
import pandas as pd



RAW_DIR = Path("data/raw/exogenous")
RAW_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_DAYS = 30

# Mismo rango temporal que el precio
DEFAULT_START = "2020-01-01T00:00"
DEFAULT_END = "2024-12-31T23:59"


# 460  -> Previsión diaria de la demanda eléctrica peninsular
# 541  -> Previsión de la producción eólica nacional peninsular
# 542  -> Generación prevista solar fotovoltaica
# 10063 -> Generación programada P48 UGH + no UGH (hidráulica agregada)
INDICATORS = {
    460: "demand_forecast",
    541: "wind_forecast",
    542: "solar_forecast",
    10063: "hydro_programmed",
}



def parse_iso_local(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M")


def to_iso_local(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M")


def esios_get(token: str, indicator_id: int, params: dict) -> dict:
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"
    headers = {
        "Accept": "application/json; application/vnd.esios-api-v2+json",
        "Content-Type": "application/json",
        "x-api-key": token,
        "Authorization": f"Token token={token}",
    }
    r = requests.get(url, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def values_to_df(payload: dict, value_name: str) -> pd.DataFrame:
    """
    Convierte la respuesta de ESIOS a un DataFrame homogéneo.
    """
    values = payload.get("indicator", {}).get("values", [])
    if not values:
        return pd.DataFrame(columns=["timestamp_utc", value_name])

    df = pd.json_normalize(values)

    
    df = df.rename(columns={"datetime_utc": "timestamp_utc", "value": value_name})

    if "timestamp_utc" not in df.columns or value_name not in df.columns:
        return pd.DataFrame(columns=["timestamp_utc", value_name])

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")

    df = df[["timestamp_utc", value_name]].dropna()
    df = df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"]).reset_index(drop=True)

    return df


def download_one_indicator(
    token: str,
    indicator_id: int,
    value_name: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Descarga un indicador en chunks temporales y devuelve un DataFrame consolidado.
    """
    print(f"\nDescargando indicador {indicator_id} -> {value_name}")
    all_chunks = []

    cur = start_dt
    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS), end_dt)

        params = {
            "start_date": to_iso_local(cur),
            "end_date": to_iso_local(chunk_end),
            "time_trunc": "hour",
        }

        payload = esios_get(token, indicator_id, params)
        df_chunk = values_to_df(payload, value_name)

        print(f"- {to_iso_local(cur)} -> {to_iso_local(chunk_end)} | filas: {len(df_chunk)}")

        all_chunks.append(df_chunk)
        cur = chunk_end

    if not all_chunks:
        return pd.DataFrame(columns=["timestamp_utc", value_name])

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    return df



def main():
    load_dotenv()

    token = os.getenv("ESIOS_TOKEN")
    if not token:
        raise SystemExit("Falta ESIOS_TOKEN en .env")

    start = os.getenv("ESIOS_START", DEFAULT_START)
    end = os.getenv("ESIOS_END", DEFAULT_END)

    start_dt = parse_iso_local(start)
    end_dt = parse_iso_local(end)

    downloaded = []

    for indicator_id, value_name in INDICATORS.items():
        df = download_one_indicator(token, indicator_id, value_name, start_dt, end_dt)

        parquet_path = RAW_DIR / f"{indicator_id}_{value_name}.parquet"
        csv_path = RAW_DIR / f"{indicator_id}_{value_name}.csv"

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

        print(f"Guardado: {parquet_path}")
        print(f"Guardado: {csv_path}")

        if not df.empty:
            print(
                f"Filas: {len(df)} | "
                f"Desde: {df['timestamp_utc'].min()} | "
                f"Hasta: {df['timestamp_utc'].max()}"
            )
            print(df.head())

        downloaded.append(df)


    merged = None
    for df in downloaded:
        if merged is None:
            merged = df.copy()
        else:
            merged = merged.merge(df, on="timestamp_utc", how="outer")

    if merged is None:
        raise SystemExit("No se ha descargado ningún indicador.")

    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)

    merged_parquet = RAW_DIR / "exogenous_merged.parquet"
    merged_csv = RAW_DIR / "exogenous_merged.csv"

    merged.to_parquet(merged_parquet, index=False)
    merged.to_csv(merged_csv, index=False)

    print("\nOK. Dataset exógeno unificado guardado:")
    print(merged_parquet)
    print(merged_csv)
    print(f"Shape final: {merged.shape}")
    print(merged.head())


if __name__ == "__main__":
    main()