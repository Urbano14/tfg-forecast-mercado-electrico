from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.features.build_features import load_series, build_features
from src.modelos.linear_model import LinearRegressionModel
from src.evaluation.backtesting_with_time import rolling_origin_backtest_with_time


RESULTS_PATH = Path("results/linear_model_backtest_results.csv")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def split_featured_data(df_feat: pd.DataFrame):
    """
    Divide el dataset con features en train / validation / test
    respetando la temporalidad.
    """
    train = df_feat[
        (df_feat["timestamp"].dt.year >= 2020) &
        (df_feat["timestamp"].dt.year <= 2022)
    ]

    val = df_feat[df_feat["timestamp"].dt.year == 2023]
    test = df_feat[df_feat["timestamp"].dt.year == 2024]

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def run_one(
    model: LinearRegressionModel,
    df_split: pd.DataFrame,
    split_name: str,
    horizon: int = 24,
    stride: int = 24,
):
    """
    Ejecuta rolling origin backtesting sobre un split concreto
    (validation o test) y devuelve las métricas en formato diccionario.
    """
    res = rolling_origin_backtest_with_time(
        df=df_split[["timestamp", "price"]].copy(),
        model=model,
        horizon=horizon,
        stride=stride,
    )

    print(
        f"{split_name} | LinearRegression(backtest): "
        f"n_origins={res.n_origins} | "
        f"MAE={res.mae:.4f} | "
        f"RMSE={res.rmse:.4f}"
    )

    return {
        "dataset": split_name,
        "model": "LinearRegression(backtest)",
        "n_origins": res.n_origins,
        "horizon": res.horizon,
        "stride": res.stride,
        "mae": res.mae,
        "rmse": res.rmse,
    }


def main():
    # 1. Cargar serie temporal base
    df = load_series()

    # 2. Construir dataset con features
    df_feat = build_features(df)

    # 3. Dividir en train / validation / test
    train, val, test = split_featured_data(df_feat)

    # 4. Entrenar modelo lineal con train
    model = LinearRegressionModel()
    model.fit(train)

    # 5. Evaluar con rolling origin backtesting
    results = []
    results.append(run_one(model, val, "validation"))
    results.append(run_one(model, test, "test"))

    # 6. Guardar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()