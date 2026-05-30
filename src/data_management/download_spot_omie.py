from __future__ import annotations

from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo
import os

import pandas as pd
import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = RAW_DIR / "omie_spot_es.parquet"
OUTPUT_CSV = RAW_DIR / "omie_spot_es.csv"

DEFAULT_START = "2025-01-01"
DEFAULT_END = "2026-05-01"
DEFAULT_PROBE_DAY = "2025-01-01"

OMIE_TIMEZONE = ZoneInfo("Europe/Madrid")

# URL pública de descarga de OMIE para los precios del mercado diario en España.
# Si OMIE cambia el endpoint, bastará con ajustar esta constante.
OMIE_DOWNLOAD_BASE_URL = "https://www.omie.es/es/file-download"
OMIE_FILE_DIR = "marginalpdbc"


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_omie_filename(day: date, version: int = 1) -> str:
    return f"marginalpdbc_{day:%Y%m%d}.{version}"


def build_omie_url(filename: str) -> str:
    return f"{OMIE_DOWNLOAD_BASE_URL}?filename={filename}&parents={OMIE_FILE_DIR}"


def is_html_payload(response: requests.Response) -> bool:
    content_type = (response.headers.get("content-type") or "").lower()
    body_start = response.text[:200].lstrip().lower()
    return "text/html" in content_type or body_start.startswith("<!doctype html") or body_start.startswith("<html")


def fetch_omie_text(session: requests.Session, day: date) -> tuple[str, str]:
    filename = build_omie_filename(day)
    url = build_omie_url(filename)

    response = session.get(url, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"OMIE devolvió HTTP {response.status_code} para {url}")

    if is_html_payload(response):
        raise RuntimeError(f"OMIE devolvió HTML en vez del fichero esperado para {url}")

    text = response.content.decode("latin-1")
    if "MARGINALPDBC" not in text:
        raise RuntimeError(f"Contenido OMIE inesperado para {url}")

    return text, url


def parse_decimal(value: object) -> float:
    return float(str(value).strip().replace(",", "."))


def normalize_omie_rows(text: str, url: str) -> pd.DataFrame:
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.replace("\ufeff", "").strip()
        if not line or line == "*":
            continue
        cleaned_lines.append(line)

    if not cleaned_lines:
        raise RuntimeError(f"Fichero vacío o sin datos útiles: {url}")

    header = cleaned_lines[0].rstrip(";")
    if header != "MARGINALPDBC":
        raise RuntimeError(f"Cabecera OMIE inesperada en {url}: {header}")

    data_text = "\n".join(cleaned_lines[1:])
    df = pd.read_csv(
        StringIO(data_text),
        sep=";",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5],
        names=["year", "month", "day", "period", "price_pt", "price"],
        engine="python",
    )

    if df.empty:
        raise RuntimeError(f"Fichero OMIE sin filas de datos: {url}")

    for col in ["year", "month", "day", "period"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["price"] = df["price"].map(parse_decimal)

    df = df.dropna(subset=["year", "month", "day", "period", "price"])
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["period"] = df["period"].astype(int)
    df = df.sort_values("period").reset_index(drop=True)

    if df["period"].duplicated().any():
        raise RuntimeError(f"Hay periodos duplicados en el fichero OMIE: {url}")

    return df


def build_hourly_timestamps(day: date, n_hours: int) -> pd.DatetimeIndex:
    # OMIE numera periodos en hora local Europe/Madrid.
    # En días con cambio horario puede haber 23 o 25 horas.
    start = pd.Timestamp(datetime.combine(day, datetime.min.time()), tz=OMIE_TIMEZONE)
    return pd.date_range(start=start, periods=n_hours, freq="h")


def to_hourly_prices(day: date, df_day: pd.DataFrame, url: str) -> pd.DataFrame:
    max_period = int(df_day["period"].max())
    n_rows = len(df_day)
    expected_periods = list(range(1, n_rows + 1))

    if df_day["period"].tolist() != expected_periods:
        raise RuntimeError(f"Los periodos OMIE no son consecutivos en {url}")

    if max_period <= 25:
        timestamps = build_hourly_timestamps(day, n_rows)
        if len(timestamps) != n_rows:
            raise RuntimeError(f"Número de horas inesperado en {url}")

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "price": df_day["price"].to_numpy(dtype=float),
            }
        )

    if n_rows % 4 != 0:
        raise RuntimeError(f"No se puede convertir a horas un fichero con {n_rows} periodos en {url}")

    # En algunos ficheros recientes OMIE publica 96/100 periodos de 15 minutos.
    # Para mantener una serie objetivo horaria, se agregan las 4 observaciones
    # consecutivas de cada hora mediante media simple, respetando también los días
    # con 23 o 25 horas (92/100 periodos).
    hourly_prices = df_day.groupby(df_day.index // 4)["price"].mean()
    timestamps = build_hourly_timestamps(day, len(hourly_prices))

    if len(timestamps) != len(hourly_prices):
        raise RuntimeError(f"Número de horas inesperado tras agregar cuartos de hora en {url}")

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": hourly_prices.to_numpy(dtype=float),
        }
    )


def parse_omie_day(text: str, day: date, url: str) -> pd.DataFrame:
    df_day = normalize_omie_rows(text, url)

    file_dates = pd.to_datetime(df_day[["year", "month", "day"]]).dt.date.unique()
    if len(file_dates) != 1 or file_dates[0] != day:
        raise RuntimeError(f"El fichero OMIE no corresponde al día esperado {day}: {url}")

    return to_hourly_prices(day, df_day, url)


def probe_download(session: requests.Session, probe_day: date) -> None:
    print(f"Prueba OMIE: {probe_day}")
    text, url = fetch_omie_text(session, probe_day)
    df_probe = parse_omie_day(text, probe_day, url)

    print(f"OK prueba OMIE | URL: {url}")
    print(f"Filas: {len(df_probe)} | Desde: {df_probe['timestamp'].min()} | Hasta: {df_probe['timestamp'].max()}")
    print(df_probe.head())


def print_price_checks(df: pd.DataFrame) -> None:
    print("\nVerificaciones finales:")
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
    start_day = parse_iso_date(os.getenv("OMIE_START", DEFAULT_START))
    end_day = parse_iso_date(os.getenv("OMIE_END", DEFAULT_END))
    probe_day = parse_iso_date(os.getenv("OMIE_PROBE_DAY", DEFAULT_PROBE_DAY))

    if end_day < start_day:
        raise SystemExit("OMIE_END no puede ser anterior a OMIE_START")

    session = requests.Session()
    session.headers.update({"User-Agent": "TFG-Energia/omie-spot-downloader"})

    probe_download(session, probe_day)

    print(f"\nDescargando OMIE desde {start_day} hasta {end_day}")

    all_days: list[pd.DataFrame] = []
    failed_days: list[str] = []

    current_day = start_day
    while current_day <= end_day:
        try:
            text, url = fetch_omie_text(session, current_day)
            df_day = parse_omie_day(text, current_day, url)
            print(
                f"- {current_day} | filas horarias: {len(df_day)} | "
                f"{df_day['timestamp'].min()} -> {df_day['timestamp'].max()}"
            )
            all_days.append(df_day)
        except Exception as exc:
            failed_days.append(current_day.isoformat())
            print(f"WARNING: fallo al descargar/parsing {current_day}: {exc}")

        current_day += timedelta(days=1)

    if not all_days:
        raise RuntimeError("No se ha podido descargar ningún día desde OMIE.")

    df = pd.concat(all_days, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nOK. Precio OMIE guardado en:")
    print("-", OUTPUT_PARQUET)
    print("-", OUTPUT_CSV)
    print_price_checks(df)

    if failed_days:
        print("\nDías fallidos:")
        for failed_day in failed_days:
            print("-", failed_day)
    else:
        print("\nNo hubo días fallidos.")


if __name__ == "__main__":
    main()
