from __future__ import annotations

from pathlib import Path
import pandas as pd

# AutoGluon TimeSeries proporciona el formato de datos y el predictor que permite usar Chronos-2.
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


DATA_PATH = Path("data/processed/spot_es_with_exogenous.parquet")

RESULTS_PATH = Path("results/chronos2_with_covariates_results.csv")

# Carpeta donde AutoGluon guarda el predictor y todos los artefactos del modelo.
# Chronos/AutoGluon no se guarda como un .pkl simple, sino como una carpeta completa, se usa en backend.
MODEL_DIR = Path("models/chronos2_with_covariates")


def load_data() -> pd.DataFrame:
    
    df = pd.read_parquet(DATA_PATH)

    # Convierte timestamp a datetime y elimina la zona horaria explícita.
    # AutoGluon TimeSeries trabaja mejor con timestamps sin timezone.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# Adapta el dataset al formato que necesita AutoGluon TimeSeries.
def prepare_data():
    df = load_data()

    # Variables exógenas que se pasan a AutoGluon como known_covariates.
    covariates = [
        "demand_forecast",
        "wind_forecast",
        "solar_forecast",
        "hydro_programmed",
    ]

    # Se conservan solo las columnas necesarias para Chronos-2.
    df = df[["timestamp", "price"] + covariates].copy()

    # AutoGluon espera una columna objetivo. Aquí renombramos price como target.
    df = df.rename(columns={"price": "target"})

    # AutoGluon está preparado para trabajar con varias series temporales.
    # Como solo tenemos una serie de precio español, le damos un item_id fijo.
    df["item_id"] = "precio_es"

    # Reordena columnas al formato esperado: identificador, timestamp, target y covariables.
    df = df[["item_id", "timestamp", "target"] + covariates]

   
    train_df = df[df["timestamp"].dt.year <= 2023].copy()
    test_df = df[df["timestamp"].dt.year == 2024].copy()

    # Convierte el DataFrame de entrenamiento al formato TimeSeriesDataFrame de AutoGluon.
    train_ts = TimeSeriesDataFrame.from_data_frame(
        train_df,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    # Convierte el DataFrame de test al mismo formato.
    test_ts = TimeSeriesDataFrame.from_data_frame(
        test_df,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    return train_ts, test_ts, covariates


# Ejecuta el experimento de Chronos-2 con covariables.
def main():
    print("=== Chronos-2 with covariates ===")

    # Horizonte de predicción: próximas 24 horas.
    prediction_length = 24

    # Prepara datos de entrenamiento, test y lista de covariables conocidas.
    train_ts, test_ts, covariates = prepare_data()

    # Crea el predictor de AutoGluon TimeSeries.
    # AutoGluon es el framework; Chronos-2 es el modelo configurado dentro de fit().
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target="target",
        known_covariates_names=covariates,
        eval_metric="MAE",
        path=str(MODEL_DIR),
    )

    # Entrena/evalúa Chronos-2 con dos variantes:
    # - ZeroShot: usa Chronos-2 sin fine-tuning específico.
    # - FineTuned: permite ajustar Chronos-2 con los datos del mercado eléctrico.
    predictor.fit(
        #A ambas variantes les paso la misma serie y las mismas covariables
        train_data=train_ts,
        hyperparameters={
            "Chronos2": [
                #ZeroShot usa el modelo preentrenado directamente. 
                {"ag_args": {"name_suffix": "ZeroShot"}},
                #FineTuned realiza un entrenamiento adicional sobre mis datos para adaptarse mejor.
                {
                    "fine_tune": True,
                    "ag_args": {"name_suffix": "FineTuned"},
                },
            ]
        },
        enable_ensemble=False,
        verbosity=2,
        time_limit=1800,
    )

    # Evalúa los modelos entrenados sobre el conjunto de test y genera una leaderboard.
    leaderboard = predictor.leaderboard(test_ts) #Una leaderboard es una tabla de resultados de AutoGluon.
    #Después de entrenar varios modelos o variantes, AutoGluon crea una tabla comparando su rendimiento.
    print("\n=== Leaderboard ===")
    print(leaderboard)

    # Guarda la leaderboard con los resultados en CSV.
    leaderboard.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
