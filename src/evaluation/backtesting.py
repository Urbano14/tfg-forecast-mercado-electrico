from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, rmse


class ForecastModel(Protocol):
    """
    Interfaz mínima que debe cumplir cualquier modelo para poder evaluarse
    con este motor de backtesting.

    La idea es simple:
    - el modelo recibe un histórico de precios
    - el modelo devuelve una predicción de longitud 'horizon'

    Da igual si el modelo es Naive, Seasonal Naive, lineal o XGBoost:
    mientras tenga este método forecast(...), el backtesting podrá usarlo.
    """
    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        ...
        # Cada modelo concreto implementará aquí su lógica de predicción.


@dataclass
class BacktestResult:
    """
    Objeto sencillo para devolver el resultado final del backtesting.

    Guarda:
    - cuántos orígenes de predicción se han evaluado
    - qué horizonte se usó
    - qué stride se usó
    - las métricas agregadas finales (MAE y RMSE)
    """
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
    Evalúa un modelo sobre una serie temporal usando rolling origin backtesting.

    Idea general:
    - tomamos un punto t como "momento actual"
    - damos al modelo todo lo anterior a t como histórico
    - le pedimos que prediga las próximas 'horizon' horas
    - comparamos con lo que realmente pasó
    - avanzamos el origen y repetimos

    Esto simula el uso real del modelo en forecasting:
    en cada momento solo conoce el pasado, nunca el futuro.
    """


    y = series.astype(float).to_numpy() 

    
    n = len(y)
    if n < horizon + 10:
        raise ValueError(f"Serie demasiado corta ({n}) para horizon={horizon}")

    # Si no se especifica desde dónde empezar, empezamos tras una semana completa.
    if start_index is None:
        start_index = 24 * 7

    # Si no se especifica dónde terminar, llegamos hasta el final de la serie
    # menos el horizonte, para asegurarnos de que siempre haya valores reales
    # con los que comparar la predicción.
    if end_index is None:
        end_index = n - horizon

    # Validaciones para evitar rangos inconsistentes.
    if start_index < 1:
        raise ValueError("start_index debe ser >= 1")

    if end_index > n - horizon:
        end_index = n - horizon

    if start_index >= end_index:
        raise ValueError(
            f"Rango inválido: start_index={start_index}, end_index={end_index}"
        )

    # Acumuladores para guardar:
    # - todos los valores reales futuros observados
    # - todas las predicciones del modelo
    # para luego calcular métricas globales.
    all_true = []
    all_pred = []

    # Contador de cuántos exámenes hemos hecho.
    origins = 0

    # Primer origen temporal desde el que empezamos a predecir.
    t = start_index
  
    # En cada iteración:
    # - history = pasado disponible hasta t
    # - y_true  = futuro real de longitud horizon
    # - y_pred  = predicción del modelo para ese futuro
    while t < end_index:
        # Todo lo anterior a t es el histórico que el modelo puede usar.
        history = y[:t]

        # Los siguientes 'horizon' puntos son lo que queremos predecir.
        y_true = y[t : t + horizon]

        # El modelo hace su forecast usando solo el histórico.
        y_pred = model.forecast(history=history, horizon=horizon)

        y_pred = np.asarray(y_pred, dtype=float)

        
        if y_pred.shape != (horizon,):
            raise ValueError(
                f"El modelo devolvió shape {y_pred.shape}, esperado {(horizon,)}"
            )

        # Guardamos esta predicción y su verdad correspondiente.
        all_true.append(y_true)
        all_pred.append(y_pred)

        # Contamos un origen más evaluado.
        origins += 1

        # Avanzamos el origen.
        t += stride

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)

    # Calculamos métricas globales sobre todas las predicciones acumuladas
    # y devolvemos el resultado final del backtesting.
    return BacktestResult(
        n_origins=origins,
        horizon=horizon,
        stride=stride,
        mae=mae(y_true_all, y_pred_all),
        rmse=rmse(y_true_all, y_pred_all),
    )

