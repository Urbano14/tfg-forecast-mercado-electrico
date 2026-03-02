from __future__ import annotations
import numpy as np


def mae(y_true, y_pred) -> float:
    """
    Mean Absolute Error (MAE).
    Devuelve el error medio absoluto en las mismas unidades que la variable (€/MWh).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"mae: shape mismatch {y_true.shape} vs {y_pred.shape}")

    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """
    Root Mean Squared Error (RMSE).
    Penaliza más los errores grandes (por el cuadrado).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"rmse: shape mismatch {y_true.shape} vs {y_pred.shape}")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


if __name__ == "__main__":
    # Ejemplo de uso
    yt = [1, 2, 3]
    yp = [2, 2, 4]
    print("MAE:", mae(yt, yp))   
    print("RMSE:", rmse(yt, yp)) 