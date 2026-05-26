from pathlib import Path
import json

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# Importa funciones de build_windows para cargar los datos y preparar las ventanas de entrenamiento.
from src.features.build_windows import (
    CALENDAR_COLS,
    DEFAULT_HORIZON,
    DEFAULT_INPUT_WINDOW,
    EXOGENOUS_COLS,
    build_supervised_windows,
    load_series,
)


MODELS_DIR = Path("models/xgboost")
RESULTS_DIR = Path("results/xgboost_multistep")
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 20  # 50 tardaban mucho.

# Por defecto no se reentrena XGBoost ni se vuelve a ejecutar Optuna.
# Cambiar a True solo cuando se quieran regenerar los .pkl y las métricas finales.
RUN_FULL_XGBOOST_PIPELINE = False

BASE_XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Tipos auxiliares para que las funciones sean más legibles.
WindowData = tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]
Splits = dict[str, WindowData]


def ensure_dirs() -> None:
    # Crea las carpetas donde se guardan modelos y resultados.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_model(params: dict | None = None) -> MultiOutputRegressor:
    # Crea un XGBoost preparado para salida múltiple.
    # Como y tiene 24 columnas, MultiOutputRegressor permite predecir las 24 horas.
    model_params = dict(BASE_XGBOOST_PARAMS)
    if params is not None:
        model_params.update(params)

    return MultiOutputRegressor(
        XGBRegressor(**model_params)
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict | None = None,
) -> MultiOutputRegressor:
    # Entrena un modelo XGBoost multi-output con X_train e y_train.
    model = make_model(params)
    model.fit(X_train, y_train)
    return model


def suggest_xgboost_params(trial: optuna.Trial) -> dict:
    # Define el espacio de búsqueda de Optuna para los hiperparámetros de XGBoost.
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
    }


# Sirve para evaluar predicciones de 24 horas.
def evaluate_multistep(
    y_true: np.ndarray,  # Matriz (N, 24) con los valores reales.
    y_pred: np.ndarray,  # Matriz (N, 24) con los valores predichos.
) -> tuple[dict[str, float], pd.DataFrame]:

    # Verifica que las formas de y_true y y_pred sean compatibles.
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # Verifica que sean matrices 2D: filas = muestras, columnas = horizonte.
    if y_true.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays")

    # Aplana ambas matrices y calcula las métricas globales considerando todas las horas predichas.
    mae_global = float(mean_absolute_error(y_true.ravel(), y_pred.ravel()))
    rmse_global = float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())))

    rows = []
    for h in range(y_true.shape[1]):  # Calcula MAE/RMSE para cada hora del horizonte, de la 1 a la 24.
        rows.append(
            {
                "horizon": h + 1,
                "mae": float(mean_absolute_error(y_true[:, h], y_pred[:, h])),
                "rmse": float(np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))),
            }
        )

    metrics_summary = {
        "mae_global": mae_global,
        "rmse_global": rmse_global,
        "horizon": y_true.shape[1],
        "n_samples": y_true.shape[0],
    }
    metrics_by_horizon = pd.DataFrame(rows)  # Tabla con métricas por cada hora del horizonte.
    return metrics_summary, metrics_by_horizon


# Guarda las métricas calculadas por evaluate_multistep.
def save_metrics(
    summary: dict[str, float],  # Resumen global.
    by_horizon: pd.DataFrame,  # Tabla con métricas hora por hora.
    variant: str,
    split: str,
) -> None:
    summary_path = RESULTS_DIR / f"{variant}_{split}_metrics_summary.json"
    by_horizon_path = RESULTS_DIR / f"{variant}_{split}_metrics_by_horizon.csv"

    serializable_summary = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in summary.items()
    }

    # El resumen global se guarda en JSON porque es información clave-valor.
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2)

    # Las métricas por horizonte se guardan en CSV porque son una tabla.
    by_horizon.to_csv(by_horizon_path, index=False)


# Seasonal Naive adaptado al formato de ventanas.
# Cada fila de X empieza con 168 precios pasados, así que se usan las últimas 24 horas observadas como predicción.
def seasonal_naive_predictions_from_windows(
    X: np.ndarray,
    input_window: int = DEFAULT_INPUT_WINDOW,
    horizon: int = DEFAULT_HORIZON,
) -> np.ndarray:
    if input_window < horizon:
        raise ValueError("input_window must be greater than or equal to horizon")

    y_pred = X[:, input_window - horizon : input_window]
    if y_pred.shape != (X.shape[0], horizon):
        raise ValueError("Seasonal naive predictions must have shape (n_samples, horizon)")
    return y_pred


# Optimiza los hiperparámetros usando Optuna y las funciones auxiliares anteriores.
def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
    study_name: str = "xgboost_multistep",
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        # En cada trial, Optuna propone parámetros, entrena un modelo y evalúa en validación.
        params = suggest_xgboost_params(trial)
        model = train_model(X_train, y_train, params)
        y_pred = model.predict(X_val)
        metrics_summary, _ = evaluate_multistep(y_val, y_pred)
        return metrics_summary["mae_global"]  # Optuna minimiza el MAE global de validación.

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials)
    return study


# Divide X, y y timestamps por años: train hasta 2022, validación 2023 y test 2024.
def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> Splits:
    train_mask = timestamps.year <= 2022
    val_mask = timestamps.year == 2023
    test_mask = timestamps.year == 2024

    return {
        "train": (X[train_mask], y[train_mask], timestamps[train_mask]),
        "val": (X[val_mask], y[val_mask], timestamps[val_mask]),
        "test": (X[test_mask], y[test_mask], timestamps[test_mask]),
    }


def print_split_summary(name: str, splits: Splits) -> None:
    # Muestra por consola las dimensiones y el rango temporal de cada split.
    print(f"{name} splits:")
    for split_name, (X_split, y_split, ts_split) in splits.items():
        print(
            f"{split_name}: X={X_split.shape}, y={y_split.shape}, "
            f"range={ts_split[0]} -> {ts_split[-1]}"
        )


def format_split_shapes(splits: Splits) -> str:
    # Devuelve un texto corto con las dimensiones de train, val y test.
    train_shape = splits["train"][0].shape
    val_shape = splits["val"][0].shape
    test_shape = splits["test"][0].shape
    return f"train={train_shape}, val={val_shape}, test={test_shape}"


# Carga el dataset final y muestra el rango temporal.
def prepare_dataset() -> pd.DataFrame:
    df = load_series()  # Carga spot_es_with_exogenous.parquet y añade calendario.

    # Primera y última fecha del dataset para comprobar el rango temporal.
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()

    print(f"Dataset loaded: {len(df)} rows | {start_ts} -> {end_ts}")
    print(f"Window: {DEFAULT_INPUT_WINDOW} | Horizon: {DEFAULT_HORIZON}")

    return df


# Construye las tres variantes de ventanas: base, mínima y completa.
def build_window_variants(df: pd.DataFrame) -> dict[str, WindowData]:
    # Variante base: solo 168 precios pasados. y = 24 precios futuros reales.
    X_base, y_base, timestamps_base = build_supervised_windows(df)

    # Variante mínima: 168 precios pasados + variables calendario futuras.
    X_minimal, y_minimal, timestamps_minimal = build_supervised_windows(df,future_covariate_cols=CALENDAR_COLS,)

    # Variante completa: 168 precios pasados + calendario futuro + exógenas futuras.
    X_complete, y_complete, timestamps_complete = build_supervised_windows(df,future_covariate_cols=[*CALENDAR_COLS, *EXOGENOUS_COLS],)

    return {
        "base": (X_base, y_base, timestamps_base),
        "minimal": (X_minimal, y_minimal, timestamps_minimal),
        "complete": (X_complete, y_complete, timestamps_complete),
    }


# Comprueba que las variantes tengan las dimensiones esperadas.
def validate_window_variants(variants: dict[str, WindowData]) -> int:
    X_base, y_base, timestamps_base = variants["base"]
    X_minimal, y_minimal, timestamps_minimal = variants["minimal"]
    X_complete, y_complete, timestamps_complete = variants["complete"]

    # Comprueba que el número de filas de X, y y timestamps sea el mismo.
    assert X_base.shape[0] == y_base.shape[0] == len(timestamps_base)
    assert X_base.shape[1] == DEFAULT_INPUT_WINDOW  # 168 columnas = 168 precios pasados.
    assert y_base.shape[1] == DEFAULT_HORIZON  # 24 columnas = 24 precios futuros.

    assert X_minimal.shape[0] == y_minimal.shape[0] == len(timestamps_minimal)
    assert y_minimal.shape[1] == DEFAULT_HORIZON
    # 168 precios pasados + 7 variables calendario * 24 horas = 336 columnas.
    assert X_minimal.shape[1] == DEFAULT_INPUT_WINDOW + len(CALENDAR_COLS) * DEFAULT_HORIZON
    assert np.array_equal(y_base, y_minimal)
    assert timestamps_base.equals(timestamps_minimal)

    assert X_complete.shape[0] == y_complete.shape[0] == len(timestamps_complete)
    assert y_complete.shape[1] == DEFAULT_HORIZON
    # 168 precios pasados + (7 calendario + 4 exógenas) * 24 horas = 432 columnas.
    assert X_complete.shape[1] == (
        DEFAULT_INPUT_WINDOW
        + (len(CALENDAR_COLS) + len(EXOGENOUS_COLS)) * DEFAULT_HORIZON
    )

    # Muestras perdidas por exigir exógenas futuras completas.
    return len(timestamps_minimal) - len(timestamps_complete)


# Hace el split temporal para las variantes mínima y completa.
def split_window_variants(variants: dict[str, WindowData]) -> dict[str, Splits]:
    X_minimal, y_minimal, timestamps_minimal = variants["minimal"]
    X_complete, y_complete, timestamps_complete = variants["complete"]

    return {
        "minimal": temporal_split(X_minimal, y_minimal, timestamps_minimal),
        "complete": temporal_split(X_complete, y_complete, timestamps_complete),
    }


# Comprueba que cada split tenga dimensiones correctas y años correctos.
def validate_splits(all_splits: dict[str, Splits]) -> None:
    for splits in all_splits.values():
        for split_name, (X_split, y_split, ts_split) in splits.items():
            assert X_split.shape[0] == y_split.shape[0] == len(ts_split)
            assert y_split.shape[1] == DEFAULT_HORIZON

            if split_name == "train":
                assert (ts_split.year <= 2022).all()
            elif split_name == "val":
                assert (ts_split.year == 2023).all()
            elif split_name == "test":
                assert (ts_split.year == 2024).all()


# Muestra por pantalla el tamaño de las variantes preparadas.
def print_prepared_variants(all_splits: dict[str, Splits], lost_complete_samples: int) -> None:
    print()
    print("Prepared variants:")
    print(f"minimal: {format_split_shapes(all_splits['minimal'])}")
    print(
        f"complete: {format_split_shapes(all_splits['complete'])}, "
        f"lost_samples={lost_complete_samples}"
    )


# Evalúa Seasonal Naive usando las ventanas del enfoque multi-step.
def evaluate_seasonal_naive(minimal_splits: Splits) -> None:
    # Se usa la variante mínima porque sus primeras 168 columnas son los precios pasados.
    X_val_min, y_val_min, _ = minimal_splits["val"]
    X_min_test, y_min_test, _ = minimal_splits["test"]

    # Predice copiando las últimas 24 horas observadas de cada ventana.
    y_seasonal_val_pred = seasonal_naive_predictions_from_windows(X_val_min)
    y_seasonal_test_pred = seasonal_naive_predictions_from_windows(X_min_test)

    # Calcula métricas globales y por horizonte.
    seasonal_val_summary, seasonal_val_by_horizon = evaluate_multistep(
        y_val_min,
        y_seasonal_val_pred,
    )
    seasonal_test_summary, seasonal_test_by_horizon = evaluate_multistep(
        y_min_test,
        y_seasonal_test_pred,
    )

    # Guarda las métricas de validación y test.
    save_metrics(seasonal_val_summary, seasonal_val_by_horizon, "seasonal_naive", "val")
    save_metrics(seasonal_test_summary, seasonal_test_by_horizon, "seasonal_naive", "test")

    print()
    print("Seasonal Naive metrics:")
    print(
        f"val: MAE={seasonal_val_summary['mae_global']}, "
        f"RMSE={seasonal_val_summary['rmse_global']}"
    )
    print(
        f"test: MAE={seasonal_test_summary['mae_global']}, "
        f"RMSE={seasonal_test_summary['rmse_global']}"
    )


# Ejecuta el pipeline completo para una variante XGBoost: minimal o complete.
def run_xgboost_variant(variant: str, splits: Splits) -> None:
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    X_test, y_test, _ = splits["test"]

    print(f"\nOptimizing XGBoost {variant}...")
    study = optimize_xgboost(
        X_train,
        y_train,
        X_val,
        y_val,
        study_name=f"xgboost_multistep_{variant}",
    )

    # Guarda los mejores hiperparámetros encontrados por Optuna.
    best_params_path = RESULTS_DIR / f"{variant}_best_params.json"
    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    print(f"Training final XGBoost {variant} with best params...")
    model = train_model(X_train, y_train, study.best_params)

    # Guarda el modelo entrenado que luego puede cargar el backend.
    model_path = MODELS_DIR / f"xgboost_multistep_{variant}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved model: {model_path}")

    # Evalúa el modelo en validación y test.
    for split_name, X_split, y_split in (
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        y_pred = model.predict(X_split)
        summary, by_horizon = evaluate_multistep(y_split, y_pred)
        save_metrics(summary, by_horizon, variant, split_name)
        print(
            f"{variant} {split_name}: "
            f"MAE={summary['mae_global']}, RMSE={summary['rmse_global']}"
        )


# Ejecuta el entrenamiento completo de XGBoost para las dos variantes.
def run_full_xgboost_pipeline(all_splits: dict[str, Splits]) -> None:
    run_xgboost_variant("minimal", all_splits["minimal"])
    run_xgboost_variant("complete", all_splits["complete"])


# Mensaje para dejar claro que por defecto no se reentrena XGBoost.
def print_xgboost_skip_message() -> None:
    print()
    print("XGBoost multi-step training skipped.")
    print("Set RUN_FULL_XGBOOST_PIPELINE = True to regenerate:")
    print(f"- {MODELS_DIR / 'xgboost_multistep_minimal.pkl'}")
    print(f"- {MODELS_DIR / 'xgboost_multistep_complete.pkl'}")


def main() -> None:
    ensure_dirs()  # Crea las carpetas de modelos y resultados.

    df = prepare_dataset()  # Carga el dataset final.

    variants = build_window_variants(df)  # Construye base, minimal y complete.
    lost_complete_samples = validate_window_variants(variants)  # Valida dimensiones.

    all_splits = split_window_variants(variants)  # Divide minimal y complete en train/val/test.
    validate_splits(all_splits)  # Comprueba años y dimensiones de cada split.
    print_prepared_variants(all_splits, lost_complete_samples)  # Muestra resumen de variantes.

    evaluate_seasonal_naive(all_splits["minimal"])  # Evalúa Seasonal Naive en formato multi-step.

    if RUN_FULL_XGBOOST_PIPELINE:
        run_full_xgboost_pipeline(all_splits)  # Entrena, evalúa y guarda XGBoost minimal y complete.
    else:
        print_xgboost_skip_message()  # Evita reentrenar por defecto.


if __name__ == "__main__":
    main()
