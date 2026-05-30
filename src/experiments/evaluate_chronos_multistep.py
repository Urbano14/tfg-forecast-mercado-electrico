from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
except ImportError:  # pragma: no cover - depende del entorno de ejecución.
    TimeSeriesDataFrame = None
    TimeSeriesPredictor = None


DATA_PATH = Path("data/processed/spot_es_with_exogenous.parquet")
MODEL_DIR = Path("models/chronos2_with_covariates")
RESULTS_DIR = Path("results/chronos_multistep")

ITEM_ID = "ES"
INPUT_WINDOW = 168
HORIZON = 24
# En validación se usa una muestra para evitar repetir una ejecución muy larga.
MAX_WINDOWS_VAL = 500
# En test se evalúan todas las ventanas disponibles cuando el valor es None.
MAX_WINDOWS_TEST = None
PROGRESS_EVERY = 10

COVARIATE_COLS = [
    "demand_forecast",
    "wind_forecast",
    "solar_forecast",
    "hydro_programmed",
]


# Convierte la columna temporal al huso Europe/Madrid para hacer el split y guardar resultados.
def ensure_europe_madrid_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        return ts.dt.tz_localize("Europe/Madrid")
    return ts.dt.tz_convert("Europe/Madrid")


# Reproduce la normalización temporal usada al entrenar Chronos:
# timestamp en UTC y sin timezone explícita para AutoGluon.
def to_autogluon_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def ensure_dependencies() -> None:
    if TimeSeriesDataFrame is None or TimeSeriesPredictor is None:
        raise ImportError(
            "No se pudo importar autogluon.timeseries. "
            "Ejecuta este script con el entorno del proyecto donde Chronos esté instalado."
        )


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = ensure_europe_madrid_timestamp(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required_cols = ["timestamp", "price", *COVARIATE_COLS]
    df = df[required_cols].dropna().reset_index(drop=True)

    print(
        f"Dataset cargado: {len(df)} filas limpias | "
        f"{df['timestamp'].min()} -> {df['timestamp'].max()}"
    )
    return df


def validate_predictor(predictor: TimeSeriesPredictor) -> None:
    if predictor.prediction_length != HORIZON:
        raise ValueError(
            f"El predictor cargado tiene prediction_length={predictor.prediction_length}, "
            f"pero este script exige horizon={HORIZON}."
        )

    if predictor.target != "target":
        raise ValueError(
            f"El predictor cargado usa target='{predictor.target}', se esperaba 'target'."
        )

    expected_covariates = list(COVARIATE_COLS)
    loaded_covariates = list(predictor.known_covariates_names or [])
    if loaded_covariates != expected_covariates:
        raise ValueError(
            "Las known_covariates del predictor no coinciden con las esperadas.\n"
            f"Esperadas: {expected_covariates}\n"
            f"Recibidas: {loaded_covariates}"
        )


def build_context_tsdf(context_df: pd.DataFrame) -> TimeSeriesDataFrame:
    history = context_df.copy()
    history["timestamp"] = to_autogluon_timestamp(history["timestamp"])
    history["item_id"] = ITEM_ID
    history = history.rename(columns={"price": "target"})
    history = history[["item_id", "timestamp", "target", *COVARIATE_COLS]]
    return TimeSeriesDataFrame.from_data_frame(
        history,
        id_column="item_id",
        timestamp_column="timestamp",
    )


def build_known_covariates_tsdf(future_df: pd.DataFrame) -> TimeSeriesDataFrame:
    known_covariates = future_df.copy()
    known_covariates["timestamp"] = to_autogluon_timestamp(known_covariates["timestamp"])
    known_covariates["item_id"] = ITEM_ID
    known_covariates = known_covariates[["item_id", "timestamp", *COVARIATE_COLS]]
    return TimeSeriesDataFrame.from_data_frame(
        known_covariates,
        id_column="item_id",
        timestamp_column="timestamp",
    )


def candidate_window_starts(df: pd.DataFrame, split: str) -> list[int]:
    indices: list[int] = []
    max_start = len(df) - HORIZON

    for start_idx in range(INPUT_WINDOW, max_start + 1):
        forecast_start = df.iloc[start_idx]["timestamp"]
        if split == "val" and forecast_start.year == 2024:
            indices.append(start_idx)
        elif split == "test" and forecast_start.year >= 2025:
            indices.append(start_idx)

    return indices


def extract_point_forecast(predictions: TimeSeriesDataFrame) -> np.ndarray:
    pred_df = predictions.reset_index()
    if "mean" not in pred_df.columns:
        raise KeyError(
            "La salida de predictor.predict() no contiene la columna 'mean'. "
            f"Columnas recibidas: {pred_df.columns.tolist()}"
        )

    pred_df = pred_df.sort_values("timestamp").reset_index(drop=True)
    y_pred = pred_df["mean"].to_numpy(dtype=float)

    if len(y_pred) != HORIZON:
        raise ValueError(
            f"Chronos devolvió {len(y_pred)} pasos, pero se esperaban {HORIZON}."
        )
    return y_pred


def evaluate_multistep(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays")

    mae_global = float(mean_absolute_error(y_true.ravel(), y_pred.ravel()))
    rmse_global = float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())))

    rows = []
    for horizon_idx in range(y_true.shape[1]):
        rows.append(
            {
                "horizon": horizon_idx + 1,
                "mae": float(mean_absolute_error(y_true[:, horizon_idx], y_pred[:, horizon_idx])),
                "rmse": float(
                    np.sqrt(mean_squared_error(y_true[:, horizon_idx], y_pred[:, horizon_idx]))
                ),
            }
        )

    summary = {
        "mae_global": mae_global,
        "rmse_global": rmse_global,
        "horizon": int(y_true.shape[1]),
        "n_samples": int(y_true.shape[0]),
        "n_predictions": int(y_true.size),
    }
    return summary, pd.DataFrame(rows)


def evaluate_split(
    df: pd.DataFrame,
    predictor: TimeSeriesPredictor,
    split: str,
    max_windows: int | None,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    start_indices = candidate_window_starts(df, split)
    selected_indices = start_indices[:max_windows]

    if not selected_indices:
        raise ValueError(f"No se encontraron ventanas válidas para el split '{split}'.")

    print(
        f"Evaluando split={split} | ventanas candidatas={len(start_indices)} | "
        f"ventanas seleccionadas={len(selected_indices)}"
    )

    prediction_rows: list[dict[str, object]] = []
    y_true_windows: list[np.ndarray] = []
    y_pred_windows: list[np.ndarray] = []
    successful_window_starts: list[pd.Timestamp] = []
    successful_window_ends: list[pd.Timestamp] = []
    failed_windows = 0
    model_name = predictor.model_best

    for window_number, start_idx in enumerate(selected_indices, start=1):
        context_df = df.iloc[start_idx - INPUT_WINDOW : start_idx].copy()
        future_df = df.iloc[start_idx : start_idx + HORIZON].copy()

        if len(context_df) != INPUT_WINDOW or len(future_df) != HORIZON:
            warnings.warn(
                f"[{split}] Ventana {window_number} omitida por longitud inesperada "
                f"(contexto={len(context_df)}, futuro={len(future_df)})."
            )
            failed_windows += 1
            continue

        try:
            history_ts = build_context_tsdf(context_df)
            known_covariates_ts = build_known_covariates_tsdf(future_df)
            predictions = predictor.predict(
                history_ts,
                known_covariates=known_covariates_ts,
                model=model_name,
            )
            y_pred = extract_point_forecast(predictions)
        except Exception as exc:  # pragma: no cover - depende del modelo/entorno.
            warnings.warn(
                f"[{split}] Ventana {window_number} falló en Chronos "
                f"(forecast_start={future_df.iloc[0]['timestamp']}): {exc}"
            )
            failed_windows += 1
            continue

        y_true = future_df["price"].to_numpy(dtype=float)
        y_true_windows.append(y_true)
        y_pred_windows.append(y_pred)
        successful_window_starts.append(future_df.iloc[0]["timestamp"])
        successful_window_ends.append(future_df.iloc[-1]["timestamp"])

        window_start_timestamp = future_df.iloc[0]["timestamp"]
        for horizon_idx in range(HORIZON):
            prediction_rows.append(
                {
                    "window_start_timestamp": window_start_timestamp,
                    "forecast_timestamp": future_df.iloc[horizon_idx]["timestamp"],
                    "horizon": horizon_idx + 1,
                    "y_true": float(y_true[horizon_idx]),
                    "y_pred": float(y_pred[horizon_idx]),
                }
            )

        if window_number % PROGRESS_EVERY == 0 or window_number == len(selected_indices):
            print(
                f"[{split}] Progreso: {window_number}/{len(selected_indices)} "
                f"ventanas procesadas | fallidas={failed_windows}"
            )

    if not y_true_windows:
        raise RuntimeError(
            f"No se pudo evaluar ninguna ventana del split '{split}'. "
            "Revisa los warnings de Chronos, las covariables y el acceso a artefactos del modelo."
        )

    y_true_matrix = np.vstack(y_true_windows)
    y_pred_matrix = np.vstack(y_pred_windows)
    summary, by_horizon = evaluate_multistep(y_true_matrix, y_pred_matrix)
    summary["split"] = split
    summary["max_windows"] = None if max_windows is None else int(max_windows)
    summary["n_candidate_windows"] = int(len(start_indices))
    summary["n_windows_ok"] = int(len(y_true_windows))
    summary["n_windows_failed"] = int(failed_windows)
    summary["start_timestamp"] = str(successful_window_starts[0])
    summary["end_timestamp"] = str(successful_window_ends[-1])

    predictions_df = pd.DataFrame(prediction_rows)
    return predictions_df, summary, by_horizon


def save_split_outputs(
    split: str,
    predictions_df: pd.DataFrame,
    summary: dict[str, float],
    by_horizon: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = RESULTS_DIR / f"chronos_{split}_predictions.csv"
    summary_path = RESULTS_DIR / f"chronos_{split}_metrics_summary.json"
    by_horizon_path = RESULTS_DIR / f"chronos_{split}_metrics_by_horizon.csv"

    predictions_df.to_csv(predictions_path, index=False)
    by_horizon.to_csv(by_horizon_path, index=False)

    serializable_summary = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in summary.items()
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2)

    print(f"Guardado: {predictions_path}")
    print(f"Guardado: {summary_path}")
    print(f"Guardado: {by_horizon_path}")


def main() -> None:
    ensure_dependencies()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Evaluación Chronos multi-step 168h -> 24h ===")
    print("Criterio temporal: train <= 2023, validation = 2024, test >= 2025.")

    df = load_dataset()

    predictor = TimeSeriesPredictor.load(str(MODEL_DIR))
    validate_predictor(predictor)

    print(f"Predictor cargado desde: {MODEL_DIR}")
    print(f"Modelo por defecto: {predictor.model_best}")
    print(f"Known covariates: {predictor.known_covariates_names}")

    val_predictions, val_summary, val_by_horizon = evaluate_split(
        df=df,
        predictor=predictor,
        split="val",
        max_windows=MAX_WINDOWS_VAL,
    )
    save_split_outputs("val", val_predictions, val_summary, val_by_horizon)

    test_predictions, test_summary, test_by_horizon = evaluate_split(
        df=df,
        predictor=predictor,
        split="test",
        max_windows=MAX_WINDOWS_TEST,
    )
    save_split_outputs("test", test_predictions, test_summary, test_by_horizon)

    print()
    print(
        f"Validation: MAE={val_summary['mae_global']:.4f} | "
        f"RMSE={val_summary['rmse_global']:.4f} | "
        f"ventanas_ok={val_summary['n_windows_ok']}"
    )
    print(
        f"Test: MAE={test_summary['mae_global']:.4f} | "
        f"RMSE={test_summary['rmse_global']:.4f} | "
        f"ventanas_ok={test_summary['n_windows_ok']}"
    )


if __name__ == "__main__":
    main()
