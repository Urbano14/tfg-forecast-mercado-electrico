from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.features.build_features import load_series, build_features
from src.modelos.linear_model import LinearRegressionModel
from src.evaluation.metrics import mae, rmse


RESULTS_PATH = Path("results/linear_model_results.csv")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def split_featured_data(df_feat: pd.DataFrame):
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

# Evalua el modelo en un conjunto concreto, por ejemplo validación o test.
def evaluate_one_split(model: LinearRegressionModel, df_split: pd.DataFrame, split_name: str):
    y_true = df_split["price"]
    y_pred = model.predict(df_split) 

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
    model.fit(train) #Aqui si entrena, no como en el modelo de baseline que no tiene entrenamiento.

    results = []
    #Aquí validacion hace lo mismo que test, no sirve para nada, pero lo dejo para que se vea el proceso completo. 
    # En un modelo más complejo, la validación se usaría para ajustar hiperparámetros o tomar decisiones de modelado,
    #mientras que el test se usaría solo para evaluar el rendimiento final.
    results.append(evaluate_one_split(model, val, "validation"))
    results.append(evaluate_one_split(model, test, "test"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()