from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression


FEATURE_COLS = [
    "lag_1",
    "lag_24",
    "lag_168",
    "demand_forecast",
    "wind_forecast",
    "solar_forecast",
    "hydro_programmed",
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
    Modelo lineal supervisado para forecasting tabular.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.feature_cols = FEATURE_COLS
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> None:
        X_train = df_train[self.feature_cols]
        y_train = df_train["price"]

        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, df_input: pd.DataFrame):
        if not self.is_fitted:
            raise ValueError("El modelo debe entrenarse antes de llamar a predict().")

        X = df_input[self.feature_cols]
        return self.model.predict(X)