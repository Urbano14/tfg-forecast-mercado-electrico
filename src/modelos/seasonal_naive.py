from __future__ import annotations
import numpy as np


class SeasonalNaiveModel:
    """
    Baseline Seasonal Naive (estacionalidad diaria):
    y_hat[t+h] = y[t+h-24]

    Para horizon=24:
      pred = [y[t-24], y[t-23], ..., y[t-1]]
    """

    def __init__(self, season_length: int = 24):
        self.season_length = int(season_length)
        if self.season_length <= 0:
            raise ValueError("season_length debe ser > 0")

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        if history.size < self.season_length:
            raise ValueError(
                f"SeasonalNaiveModel necesita >= {self.season_length} valores de histórico; "
                f"recibido {history.size}"
            )

        # Extraemos la última temporada completa del histórico
        last_season = history[-self.season_length:]

        if horizon == self.season_length:
            return last_season.copy()

        # Si el horizonte es mayor que la longitud de la temporada, repetimos la última temporada
        reps = int(np.ceil(horizon / self.season_length))
        tiled = np.tile(last_season, reps)
        return tiled[:horizon].astype(float, copy=False)