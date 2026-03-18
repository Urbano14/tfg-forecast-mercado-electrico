from pathlib import Path
import pandas as pd

from src.evaluation.metrics import mae, rmse
from src.features.build_features import load_series, build_features
from src.modelos.xgboost_model import XGBoostModel


RESULTS_PATH = Path("results/xgboost_results.csv")
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


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, dataset_name: str):
    feature_cols = [c for c in train.columns if c not in ["timestamp", "price"]]

    X_train = train[feature_cols]
    y_train = train["price"]

    X_test = test[feature_cols]
    y_test = test["price"]

    print("Features:", feature_cols)

    model = XGBoostModel()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    result = {
        "dataset": dataset_name,
        "model": "XGBoost",
        "mae": mae(y_test, preds),
        "rmse": rmse(y_test, preds),
    }

    print(
        f"{dataset_name} | XGBoost: "
        f"MAE={result['mae']:.4f} | "
        f"RMSE={result['rmse']:.4f}"
    )

    return result


def main():
    print("=== Running XGBoost ===")

    df = load_series()
    df_feat = build_features(df)

    train, val, test = split_featured_data(df_feat)

    results = []
    results.append(train_and_evaluate(train, val, "validation"))
    results.append(train_and_evaluate(train, test, "test"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()