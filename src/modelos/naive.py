from __future__ import annotations
import numpy as np


class NaiveModel:
    """
    Baseline Naive (sin estacionalidad):
    y_hat[t+h] = y[t]
    y = precio real, y_hat = precio predicho, t = momento actual, h = pasos hacia el futuro

    y[100] = 95
    y_hat[101] = 95
    y_hat[102] = 95
    y_hat[103] = 95

    Si el último precio es 95
    predicción = [95,95,95,95,95,95,95,95,95,95...(24 veces)]
    """
    
    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray: 
        #history: array de precios pasados, horizon: número de pasos a predecir (24 horas)
        history = np.asarray(history, dtype=float)
        if history.size == 0:
            raise ValueError("NaiveModel necesita history no vacío")
        last = float(history[-1]) #último precio del histórico
        return np.full(horizon, last, dtype=float) #crea un array de tamaño horizon lleno del último precio
    