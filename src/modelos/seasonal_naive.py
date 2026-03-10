from __future__ import annotations
import numpy as np


class SeasonalNaiveModel:
    """
    Baseline Seasonal Naive (estacionalidad diaria):
    y_hat[t+h] = y[t+h-24]
    y = precio real, y_hat = precio predicho, t = momento actual, h = pasos hacia el futuro

    Entonces, si queremos predecir las próximas 24 horas (horizon=24):
    y_hat[t+1] = y[t-23]
    y_hat[t+2] = y[t-22]
    ...
    y_hat[t+24] = y[t] -> La predicción para cada hora futura se basa en el precio de la misma hora del día anterior.

    Tomar las últimas 24 horas reales del histórico y copiarlas tal cual como predicción futura
    """

    def __init__(self, season_length: int = 24): 
        self.season_length = int(season_length) 
        if self.season_length <= 0:
            raise ValueError("season_length debe ser > 0")

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray: #
        # history: array de precios pasados, horizon: número de pasos a predecir (24 horas) -> devuelve un array con las predicciones
        history = np.asarray(history, dtype=float) 
        if history.size < self.season_length: 
            raise ValueError(
                f"SeasonalNaiveModel necesita >= {self.season_length} valores de histórico; "
                f"recibido {history.size}"
            )

        
        last_season = history[-self.season_length:] #last_season es un array con las últimas 24 horas del histórico, el último día completo que el modelo ha visto

        #Si el horizonte es igual a la temporada, delvolvemos las ultimas 24 horas tal cual como predicción
        if horizon == self.season_length:
            return last_season.copy() 

        #Si el horizonte es mayor que la temporada, repetimos las últimas 24 horas tantas veces como sea necesario para cubrir el horizonte,
        #  y luego recortamos el array resultante al tamaño del horizonte
        reps = int(np.ceil(horizon / self.season_length)) 
        tiled = np.tile(last_season, reps)
        return tiled[:horizon].astype(float, copy=False) 
