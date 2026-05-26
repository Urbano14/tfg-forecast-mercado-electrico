
import joblib
import optuna
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

from src.evaluation.split import temporal_split
from src.evaluation.metrics import mae, rmse
from src.features.build_features import load_series, build_features


RESULTS_PATH = Path("results/xgboost_optuna_results.csv")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path("models/xgboost/xgboost_optuna_exogenous.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

#Una prueba concreta de Optuna. Optuna llama muchas veces a objective().

def objective(trial, X_train, y_train, X_val, y_val):
    #trial.suggest_*() es la función que Optuna usa para generar valores de hiperparámetros
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "objective": "reg:squarederror",
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    return mae(y_val, preds) #Optuna intenta minimizar ese valor. Es decir: el mejor modelo será el que tenga menor MAE en validación.


def main():
    print("=== XGBoost + Optuna ===")

    df = load_series()
    df = build_features(df)

    train, val, test = temporal_split(df)

    feature_cols = [c for c in df.columns if c not in ["timestamp", "price"]]

    #X_train, X_val, X_test: lags, exógenas y calendario.
    #y_train, y_val, y_test: precio real.
    X_train = train[feature_cols]
    y_train = train["price"]

    X_val = val[feature_cols]
    y_val = val["price"]

    X_test = test[feature_cols]
    y_test = test["price"]

    #Esto crea un estudio de Optuna cuyo objetivo es minimizar una métrica. 
    #En cada llamada, propone una combinación distinta de hiperparámetros.
    study = optuna.create_study(direction="minimize")
    study.optimize( #Cada trial entrena un XGBoost con train, predice sobre validación y mide el MAE. 
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=50
        #Al final se queda con la combinación que menor MAE tenga en validación.
    )

    print("Best params:", study.best_params)

    # Generar modelo con mejores hiperparámetros
    best_model = XGBRegressor(
        **study.best_params,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    best_model.fit(X_train, y_train) #Lo entrena

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
    
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()