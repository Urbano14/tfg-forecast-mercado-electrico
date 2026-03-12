from pathlib import Path
import pandas as pd

from src.evaluation.split import load_data, temporal_split
from src.features.build_features import build_features
from src.modelos.xgboost_model import XGBoostModel
from src.evaluation.metrics import mae, rmse


RESULTS_PATH = Path("results/xgboost_results.csv")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def train_and_evaluate(train, test, dataset_name):

    df = pd.concat([train, test]).reset_index(drop=True)

    df_feat = build_features(df)

    train_feat = df_feat[df_feat["timestamp"].dt.year <= train["timestamp"].dt.year.max()]
    test_feat = df_feat[df_feat["timestamp"].dt.year == test["timestamp"].dt.year.iloc[0]]

    X_train = train_feat.drop(columns=["timestamp", "price"])
    y_train = train_feat["price"]

    print("Features:", list(X_train.columns))

    X_test = test_feat.drop(columns=["timestamp", "price"])
    y_test = test_feat["price"]

    model = XGBoostModel()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(
        f"{dataset_name} | XGBoost: "
        f"MAE={mae(y_test, preds):.4f} | "
        f"RMSE={rmse(y_test, preds):.4f}"
    )

    return {
        "dataset": dataset_name,
        "model": "XGBoost",
        "mae": mae(y_test, preds),
        "rmse": rmse(y_test, preds),
    }


def main():
    df = load_data()
    train, val, test = temporal_split(df)

    results = []

    print("=== Running XGBoost ===")

    results.append(train_and_evaluate(train, val, "validation"))
    results.append(train_and_evaluate(train, test, "test"))
    

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
    
