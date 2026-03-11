from pathlib import Path
import numpy as np
import pandas as pd


DATA_PATH = Path("data/processed/spot_es_processed.parquet")


def load_series() -> pd.DataFrame:
    """
    Carga la serie procesada y asegura orden temporal correcto.
    """
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un dataset supervisado para forecasting a partir de la serie de precios.

    Variables creadas:
    - Lags temporales
    - Variables cíclicas de hora, día de la semana y mes
    - Indicador de fin de semana
    """
    df = df.copy()

    
    # LAGS DEL PRECIO
   
    # lag_1: precio de la hora anterior
    # lag_24: precio de la misma hora del día anterior
    # lag_168: precio de la misma hora de la semana anterior
    df["lag_1"] = df["price"].shift(1)
    df["lag_24"] = df["price"].shift(24)
    df["lag_168"] = df["price"].shift(168)

    
    # VARIABLES TEMPORALES BÁSICAS
  
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek   # lunes=0, domingo=6
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    
    # VARIABLES TEMPORALES CÍCLICAS
    
    # Hora del día: ciclo de 24 horas
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Día de la semana: ciclo de 7 días
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Mes del año: ciclo de 12 meses
    # - 1 para que enero empiece en 0
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    
    # Al crear lags aparecen NaN al principio de la serie.
    # Se eliminan porque no se pueden usar para entrenar.
    df = df.dropna().reset_index(drop=True)

    
    # SELECCIÓN DE COLUMNAS
    
    cols = [
        "timestamp",
        "price",
        "lag_1",
        "lag_24",
        "lag_168",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]
    df = df[cols]

    return df


if __name__ == "__main__":
    df = load_series()
    df_feat = build_features(df)

    print(df_feat.head())
    print("\nColumnas:")
    print(df_feat.columns.tolist())
    print("\nShape:", df_feat.shape)
    print("\nRango temporal:")
    print(df_feat["timestamp"].min(), "->", df_feat["timestamp"].max())