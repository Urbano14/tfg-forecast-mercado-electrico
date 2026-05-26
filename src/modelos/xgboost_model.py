from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor


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

#no aprende una única fórmula como la regresión lineal, sino que construye muchos árboles pequeños, 
#uno detrás de otro, y cada árbol intenta corregir los errores que han dejado los anteriores.
class XGBoostModel:
    
    def __init__(self): 
        self.model = XGBRegressor(
            #Estos hiperparámetros se han elegido manuealmente, no con Optuna, eso se hace luego.
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        self.feature_cols = FEATURE_COLS
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("El modelo XGBoost no está entrenado")

        return self.model.predict(X)