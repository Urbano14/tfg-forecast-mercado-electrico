from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


FEATURE_COLS = [
    "lag_1",
    "lag_24",
    "lag_168",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


class LinearRegressionModel:
    """
    Modelo lineal para forecasting horario con predicción recursiva.

    Entrenamiento:
    - se ajusta con un dataset supervisado ya transformado en features

    Predicción:
    - para cada hora futura construye una fila de features
    - predice una hora
    - añade esa predicción al histórico extendido
    - repite hasta completar el horizonte
    """

    def __init__(self):
        self.model = LinearRegression()
        self.feature_cols = FEATURE_COLS
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Entrena el modelo con el dataframe de entrenamiento.
        """
        X_train = df_train[self.feature_cols]
        y_train = df_train["price"]

        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def _build_feature_row(
        self,
        history_extended: list[float],
        current_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Construye una única fila de features para una hora concreta.

        history_extended:
        - contiene el histórico real
        - y también las predicciones previas si estamos en forecast recursivo

        current_timestamp:
        - timestamp real de la hora que queremos predecir
        """
        lag_1 = history_extended[-1]
        lag_24 = history_extended[-24]
        lag_168 = history_extended[-168]

        hour = current_timestamp.hour
        dayofweek = current_timestamp.dayofweek
        month = current_timestamp.month

        is_weekend = 1 if dayofweek >= 5 else 0

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        dow_sin = np.sin(2 * np.pi * dayofweek / 7)
        dow_cos = np.cos(2 * np.pi * dayofweek / 7)

        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)

        return pd.DataFrame([{
            "lag_1": lag_1,
            "lag_24": lag_24,
            "lag_168": lag_168,
            "is_weekend": is_weekend,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }])

    def forecast(
        self,
        history: np.ndarray,
        start_timestamp: pd.Timestamp,
        horizon: int,
    ) -> np.ndarray:
        """
        Genera una predicción recursiva de longitud 'horizon'.

        history:
        - array con los precios históricos hasta el instante actual

        start_timestamp:
        - timestamp real de la primera hora a predecir

        horizon:
        - número de horas a predecir
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe entrenarse antes de llamar a forecast().")

        history = np.asarray(history, dtype=float)

        if history.size < 168:
            raise ValueError("Se necesitan al menos 168 valores de histórico para usar lag_168.")

        history_extended = history.tolist()
        preds = []

        for h in range(horizon):
            current_timestamp = start_timestamp + pd.Timedelta(hours=h)

            x_row = self._build_feature_row(
                history_extended=history_extended,
                current_timestamp=current_timestamp,
            )

            y_pred = float(self.model.predict(x_row)[0])

            preds.append(y_pred)
            history_extended.append(y_pred)

        return np.array(preds, dtype=float)