import optuna
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

from src.evaluation.split import load_data, temporal_split
from src.evaluation.metrics import mae, rmse
from src.features.build_features import build_features


RESULTS_PATH = Path("results/xgboost_optuna_results.csv")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def objective(trial, X_train, y_train, X_val, y_val):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = XGBRegressor(**params)

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    return mae(y_val, preds)


def main():

    print("=== XGBoost + Optuna ===")

    df = load_data()
    df = build_features(df)

    train, val, test = temporal_split(df)

    feature_cols = [c for c in df.columns if c not in ["timestamp", "price"]]

    X_train = train[feature_cols]
    y_train = train["price"]

    X_val = val[feature_cols]
    y_val = val["price"]

    X_test = test[feature_cols]
    y_test = test["price"]

    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=50
    )

    print("Best params:", study.best_params)

    best_model = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)

    best_model.fit(X_train, y_train)

    preds_val = best_model.predict(X_val)
    preds_test = best_model.predict(X_test)

    val_mae = mae(y_val, preds_val)
    val_rmse = rmse(y_val, preds_val)

    test_mae = mae(y_test, preds_test)
    test_rmse = rmse(y_test, preds_test)

    print(f"validation | XGBoost + Optuna: MAE={val_mae:.4f} RMSE={val_rmse:.4f}")
    print(f"test | XGBoost + Optuna: MAE={test_mae:.4f} RMSE={test_rmse:.4f}")

    results = pd.DataFrame([{
        "model": "XGBoost + Optuna",
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse
    }])

    results.to_csv(RESULTS_PATH, index=False)

    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()