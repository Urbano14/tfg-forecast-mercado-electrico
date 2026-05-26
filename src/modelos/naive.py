from __future__ import annotations
import numpy as np


class NaiveModel:
    
    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray: 
        #history: array de precios pasados, horizon: número de pasos a predecir (24 horas)
        history = np.asarray(history, dtype=float)
        if history.size == 0:
            raise ValueError("NaiveModel necesita history no vacío")
        last = float(history[-1]) #último precio del histórico
        return np.full(horizon, last, dtype=float) #crea un array de tamaño horizon lleno del último precio
    
#Predice que el precio de las próximas 24 sera igual al último precio observado.
#Histórico: [100, 105, 110] -> Predicción para las próximas 24 horas: [110, 110, ..., 110] (24 veces)