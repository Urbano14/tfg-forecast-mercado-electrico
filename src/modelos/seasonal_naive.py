from __future__ import annotations
import numpy as np

#Para predecir las próximas 24 horas, copia las últimas 24 horas observadas.
class SeasonalNaiveModel:
   
    def __init__(self, season_length: int = 24): 
        self.season_length = int(season_length) 
        if self.season_length <= 0:
            raise ValueError("season_length debe ser > 0")

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray: 
        # history: array de precios pasados, horizon: número de pasos a predecir (24 horas) -> devuelve un array con las predicciones
        history = np.asarray(history, dtype=float) 
        if history.size < self.season_length: 
            raise ValueError(
                f"SeasonalNaiveModel necesita >= {self.season_length} valores de histórico; "
                f"recibido {history.size}"
            )

        
        last_season = history[-self.season_length:] #Coge las últimas 24 horas del histórico.

        #Si el horizonte también es 24, devuelve esas últimas 24 horas tal cual:
        if horizon == self.season_length:
            return last_season.copy() 

        #Si el horizonte es mayor que 24, repite esas últimas 24 horas tantas veces como sea necesario
        #luego recorta el resultado para que tenga exactamente horizon elementos.
        reps = int(np.ceil(horizon / self.season_length)) 
        tiled = np.tile(last_season, reps)
        return tiled[:horizon].astype(float, copy=False) 
