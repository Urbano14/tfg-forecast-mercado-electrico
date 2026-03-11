from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.features.build_features import load_series, build_features
from src.modelos.linear_model import LinearRegressionModel
from src.evaluation.metrics import mae, rmse


RESULTS_PATH = Path("results/linear_model_results.csv")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def split_featured_data(df_feat: pd.DataFrame):
    train = df_feat[(df_feat["timestamp"].dt.year >= 2020) & (df_feat["timestamp"].dt.year <= 2022)]
    val = df_feat[df_feat["timestamp"].dt.year == 2023]
    test = df_feat[df_feat["timestamp"].dt.year == 2024]

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def evaluate_one_split(model, df_split: pd.DataFrame, split_name: str):
    """
    Evaluación simple sobre un dataframe ya transformado en features.
    Aquí no usamos aún rolling_origin_backtest porque primero queremos validar
    que el modelo lineal aprende correctamente sobre el dataset supervisado.
    """
    X = df_split[model.feature_cols]
    y_true = df_split["price"].to_numpy()
    y_pred = model.model.predict(X)

    result = {
        "dataset": split_name,
        "model": "LinearRegression",
        "n_samples": len(df_split),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }

    print(
        f"{split_name} | LinearRegression: "
        f"n_samples={len(df_split)} | "
        f"MAE={result['mae']:.4f} | "
        f"RMSE={result['rmse']:.4f}"
    )

    return result


def main():
    df = load_series()
    df_feat = build_features(df)

    train, val, test = split_featured_data(df_feat)

    model = LinearRegressionModel()
    model.fit(train)

    results = []
    results.append(evaluate_one_split(model, val, "validation"))
    results.append(evaluate_one_split(model, test, "test"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()