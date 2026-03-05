from __future__ import annotations
import numpy as np


class NaiveModel:
    """
    Baseline Naive (persistencia):
    y_hat[t+h] = y[t] para todo h=1..H
    En multi-step recursivo produce una predicción plana.
    """
    
    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        if history.size == 0:
            raise ValueError("NaiveModel necesita history no vacío")
        last = float(history[-1])
        return np.full(horizon, last, dtype=float)