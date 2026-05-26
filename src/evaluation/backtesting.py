from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, rmse


class ForecastModel(Protocol):
    #Para que un modelo pueda evaluarse con este backtesting, debe tener un método llamado forecast.
    # Este método recibe un array de precios pasados (history) y un número de pasos a predecir (horizon).
    #devuelv un array con tantas predicciones como indique horizon
    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        ...

@dataclass
class BacktestResult:
    #sirve para guardar el resultado final del backtesting
    
    n_origins: int #cuántas veces se ha movido el origen temporal y se ha hecho una predicción
    horizon: int #número de horas que se predice cada vez
    stride: int #Cuánto se avanza después de cada predicción. Si stride = 24, después de predecir 24 horas se avanza un día.
    mae: float 
    rmse: float 

#Hacer muchas predicciones de 24 horas a lo largo de la serie y calcular el error total.
def rolling_origin_backtest(
    series: pd.Series,
    model: ForecastModel,
    horizon: int = 24,
    stride: int = 24,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> BacktestResult:
    
    y = series.astype(float).to_numpy() 

    
    n = len(y) 
    if n < horizon + 10: 
        raise ValueError(f"Serie demasiado corta ({n}) para horizon={horizon}")

    # Si no se especifica desde dónde empezar, empezamos tras una semana completa.
    if start_index is None:
        start_index = 24 * 7

    # Si no se especifica dónde terminar, llegamos hasta el final de la serie
    if end_index is None:
        end_index = n - horizon

    if start_index < 1:
        raise ValueError("start_index debe ser >= 1")
    
    #Si el end_index es demasiado grande, lo ajustamos para que no se salga del rango permitido.
    if end_index > n - horizon:
        end_index = n - horizon

    if start_index >= end_index:
        raise ValueError(
            f"Rango inválido: start_index={start_index}, end_index={end_index}"
        )

   #Guardamos predicciones y verdades para calcular métricas globales al final.
    all_true = []
    all_pred = []

    # Contador de cuántos exámenes hemos hecho.
    origins = 0


    t = start_index
  
   # Hacemos predicciones desde start_index hasta end_index, avanzando stride cada vez.
    while t < end_index:
        history = y[:t] #Todos los precios anteriores a t

        y_true = y[t : t + horizon] #Valores reales futuros que queremos predecir

        y_pred = model.forecast(history=history, horizon=horizon) #Llama al modelo para que prediga el horizon a partir de history

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

    # Calculamos métricas globales sobre todas las predicciones acumuladas
    # y devolvemos el resultado final del backtesting.
    return BacktestResult(
        n_origins=origins,
        horizon=horizon,
        stride=stride,
        mae=mae(y_true_all, y_pred_all),
        rmse=rmse(y_true_all, y_pred_all),
    )

