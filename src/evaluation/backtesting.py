from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Iterable, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, rmse


class ForecastModel(Protocol):
    """
    Interfaz mínima que deben cumplir nuestros modelos para ser evaluados.

    Dado un histórico de precios (1D) devuelve una predicción de longitud horizon.
    """
    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        ...


@dataclass
class BacktestResult:
    n_origins: int
    horizon: int
    stride: int
    mae: float
    rmse: float


def rolling_origin_backtest(
    series: pd.Series,
    model: ForecastModel,
    horizon: int = 24,
    stride: int = 24,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> BacktestResult:
    """
    Backtesting tipo rolling origin sobre una serie univariante.

    - series: pd.Series indexada por timestamp o por índice (valores numéricos)
    - model: implementa forecast(history, horizon)
    - horizon: pasos a predecir (24h)
    - stride: cuánto avanzamos el origen cada iteración (24h = diario)
    - start_index/end_index: para acotar (en índice posicional). end_index es exclusivo.

    Evalúa:
      Para cada origen t:
        history = series[:t]
        y_true = series[t : t+horizon]
        y_pred = model.forecast(history, horizon)
    """
    y = series.astype(float).to_numpy()

    n = len(y)
    if n < horizon + 10:
        raise ValueError(f"Serie demasiado corta ({n}) para horizon={horizon}")

    # Por defecto: empezamos después de la primera semana (24*7=168h) para dar margen al modelo
    if start_index is None:
        start_index = 24 * 7

    # Por defecto: terminamos al final de la serie (n) menos el horizonte, para no quedarnos sin datos verdaderos que comparar
    if end_index is None:
        end_index = n - horizon

    if start_index < 1:
        raise ValueError("start_index debe ser >= 1")
    if end_index > n - horizon:
        end_index = n - horizon
    if start_index >= end_index:
        raise ValueError(f"Rango inválido: start_index={start_index}, end_index={end_index}")

    all_true = []
    all_pred = []
    origins = 0

    t = start_index
    while t < end_index:
        history = y[:t]
        y_true = y[t : t + horizon]

        y_pred = model.forecast(history=history, horizon=horizon)
        y_pred = np.asarray(y_pred, dtype=float)

        if y_pred.shape != (horizon,):
            raise ValueError(f"El modelo devolvió shape {y_pred.shape}, esperado {(horizon,)}")

        all_true.append(y_true)
        all_pred.append(y_pred)
        origins += 1

        t += stride

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)

    return BacktestResult(
        n_origins=origins,
        horizon=horizon,
        stride=stride,
        mae=mae(y_true_all, y_pred_all),
        rmse=rmse(y_true_all, y_pred_all),
    )


# Ejemplo de modelo naive: predice que el precio de las próximas 24h será igual al último precio observado
class NaiveLastValue:
    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        last = float(history[-1])
        return np.full(horizon, last, dtype=float)


if __name__ == "__main__":
    # Carga datos y backtest con modelo naive para validar que todo funciona
    from src.evaluation.split import load_data, temporal_split

    df = load_data()
    _, val, _ = temporal_split(df)
    series = val["price"]

    res = rolling_origin_backtest(series=series, model=NaiveLastValue(), horizon=24, stride=24)
    print(res)