from __future__ import annotations
import numpy as np


def mae(y_true, y_pred) -> float:
    
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"mae: shape mismatch {y_true.shape} vs {y_pred.shape}")

    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"rmse: shape mismatch {y_true.shape} vs {y_pred.shape}")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


