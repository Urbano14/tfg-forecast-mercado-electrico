from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, rmse


class TimeAwareForecastModel(Protocol):
    """
    Interfaz para modelos que necesitan conocer el timestamp real
    desde el que empieza la predicción.
    """
    def forecast(
        self,
        history: np.ndarray,
        start_timestamp: pd.Timestamp,
        horizon: int,
    ) -> np.ndarray:
        ...


@dataclass
class BacktestResult:
    n_origins: int
    horizon: int
    stride: int
    mae: float
    rmse: float


def rolling_origin_backtest_with_time(
    df: pd.DataFrame,
    model: TimeAwareForecastModel,
    horizon: int = 24,
    stride: int = 24,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> BacktestResult:
    """
    Rolling origin backtesting para modelos que necesitan timestamp real.

    El DataFrame debe contener:
    - timestamp
    - price
    """
    if "timestamp" not in df.columns or "price" not in df.columns:
        raise ValueError("El DataFrame debe contener columnas 'timestamp' y 'price'.")

    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    y = df["price"].astype(float).to_numpy()
    ts = pd.to_datetime(df["timestamp"]).reset_index(drop=True)

    n = len(y)
    if n < horizon + 10:
        raise ValueError(f"Serie demasiado corta ({n}) para horizon={horizon}")

    if start_index is None:
        start_index = 24 * 7

    if end_index is None:
        end_index = n - horizon

    if start_index < 1:
        raise ValueError("start_index debe ser >= 1")
    if end_index > n - horizon:
        end_index = n - horizon
    if start_index >= end_index:
        raise ValueError(
            f"Rango inválido: start_index={start_index}, end_index={end_index}"
        )

    all_true = []
    all_pred = []
    origins = 0

    t = start_index
    while t < end_index:
        history = y[:t]
        y_true = y[t:t + horizon]
        start_timestamp = ts.iloc[t]

        y_pred = model.forecast(
            history=history,
            start_timestamp=start_timestamp,
            horizon=horizon,
        )
        y_pred = np.asarray(y_pred, dtype=float)

        if y_pred.shape != (horizon,):
            raise ValueError(
                f"El modelo devolvió shape {y_pred.shape}, esperado {(horizon,)}"
            )

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