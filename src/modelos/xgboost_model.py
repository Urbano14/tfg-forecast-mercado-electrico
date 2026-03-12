from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class XGBoostModel:
    """
    Modelo Gradient Boosting (XGBoost) para forecasting.

    Utiliza las features generadas previamente:
    - lags
    - variables temporales

    El modelo se entrena una vez y luego se usa para predecir.
    """

    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Entrena el modelo.
        """
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones.
        """
        if not self.fitted:
            raise RuntimeError("El modelo XGBoost no está entrenado")

        return self.model.predict(X)