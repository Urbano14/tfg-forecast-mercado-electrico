from pathlib import Path
import json

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

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
N_OPTUNA_TRIALS = 20 #50 Tardaban mucho
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


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_model(params: dict | None = None) -> MultiOutputRegressor:
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
    model = make_model(params)
    model.fit(X_train, y_train)
    return model


def suggest_xgboost_params(trial: optuna.Trial) -> dict:
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
    for h in range(y_true.shape[1]):
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
    metrics_by_horizon = pd.DataFrame(rows)
    return metrics_summary, metrics_by_horizon


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


def save_metrics(
    summary: dict[str, float],
    by_horizon: pd.DataFrame,
    variant: str,
    split: str,
) -> None:
    summary_path = RESULTS_DIR / f"{variant}_{split}_metrics_summary.json"
    by_horizon_path = RESULTS_DIR / f"{variant}_{split}_metrics_by_horizon.csv"

    serializable_summary = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in summary.items()
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2)

    by_horizon.to_csv(by_horizon_path, index=False)


def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
    study_name: str = "xgboost_multistep",
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgboost_params(trial)
        model = train_model(X_train, y_train, params)
        y_pred = model.predict(X_val)
        metrics_summary, _ = evaluate_multistep(y_val, y_pred)
        return metrics_summary["mae_global"]

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    train_mask = timestamps.year <= 2022
    val_mask = timestamps.year == 2023
    test_mask = timestamps.year == 2024

    return {
        "train": (X[train_mask], y[train_mask], timestamps[train_mask]),
        "val": (X[val_mask], y[val_mask], timestamps[val_mask]),
        "test": (X[test_mask], y[test_mask], timestamps[test_mask]),
    }


def print_split_summary(
    name: str,
    splits: dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]],
) -> None:
    print(f"{name} splits:")
    for split_name, (X_split, y_split, ts_split) in splits.items():
        print(
            f"{split_name}: X={X_split.shape}, y={y_split.shape}, "
            f"range={ts_split[0]} -> {ts_split[-1]}"
        )


def format_split_shapes(
    splits: dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]],
) -> str:
    train_shape = splits["train"][0].shape
    val_shape = splits["val"][0].shape
    test_shape = splits["test"][0].shape
    return f"train={train_shape}, val={val_shape}, test={test_shape}"


def main() -> None:
    ensure_dirs()

    df = load_series()
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()

    print(f"Dataset loaded: {len(df)} rows | {start_ts} -> {end_ts}")
    print(f"Window: {DEFAULT_INPUT_WINDOW} | Horizon: {DEFAULT_HORIZON}")

    X_base, y, timestamps = build_supervised_windows(df)

    assert X_base.shape[0] == y.shape[0] == len(timestamps)
    assert X_base.shape[1] == DEFAULT_INPUT_WINDOW
    assert y.shape[1] == DEFAULT_HORIZON

    X_minimal, y_minimal, timestamps_minimal = build_supervised_windows(
        df,
        future_covariate_cols=CALENDAR_COLS,
    )

    assert X_minimal.shape[0] == y_minimal.shape[0] == len(timestamps_minimal)
    assert y_minimal.shape[1] == DEFAULT_HORIZON
    assert X_minimal.shape[1] == DEFAULT_INPUT_WINDOW + len(CALENDAR_COLS) * DEFAULT_HORIZON
    assert np.array_equal(y, y_minimal)
    assert timestamps.equals(timestamps_minimal)

    X_complete, y_complete, timestamps_complete = build_supervised_windows(
        df,
        future_covariate_cols=[*CALENDAR_COLS, *EXOGENOUS_COLS],
    )

    assert X_complete.shape[0] == y_complete.shape[0] == len(timestamps_complete)
    assert y_complete.shape[1] == DEFAULT_HORIZON
    assert X_complete.shape[1] == DEFAULT_INPUT_WINDOW + (len(CALENDAR_COLS) + len(EXOGENOUS_COLS)) * DEFAULT_HORIZON

    lost_complete_samples = len(timestamps_minimal) - len(timestamps_complete)

    minimal_splits = temporal_split(X_minimal, y_minimal, timestamps_minimal)
    complete_splits = temporal_split(X_complete, y_complete, timestamps_complete)

    for splits in (minimal_splits, complete_splits):
        for split_name, (X_split, y_split, ts_split) in splits.items():
            assert X_split.shape[0] == y_split.shape[0] == len(ts_split)
            assert y_split.shape[1] == DEFAULT_HORIZON

            if split_name == "train":
                assert (ts_split.year <= 2022).all()
            elif split_name == "val":
                assert (ts_split.year == 2023).all()
            elif split_name == "test":
                assert (ts_split.year == 2024).all()

    print()
    print("Prepared variants:")
    print(f"minimal: {format_split_shapes(minimal_splits)}")
    print(f"complete: {format_split_shapes(complete_splits)}, lost_samples={lost_complete_samples}")

    X_val_min, y_val_min, _ = minimal_splits["val"]
    X_min_test, y_min_test, _ = minimal_splits["test"]

    y_seasonal_val_pred = seasonal_naive_predictions_from_windows(X_val_min)
    y_seasonal_test_pred = seasonal_naive_predictions_from_windows(X_min_test)

    seasonal_val_summary, seasonal_val_by_horizon = evaluate_multistep(y_val_min, y_seasonal_val_pred)
    seasonal_test_summary, seasonal_test_by_horizon = evaluate_multistep(y_min_test, y_seasonal_test_pred)

    save_metrics(seasonal_val_summary, seasonal_val_by_horizon, "seasonal_naive", "val")
    save_metrics(seasonal_test_summary, seasonal_test_by_horizon, "seasonal_naive", "test")

    print()
    print("Seasonal Naive metrics:")
    print(f"val: MAE={seasonal_val_summary['mae_global']}, RMSE={seasonal_val_summary['rmse_global']}")
    print(f"test: MAE={seasonal_test_summary['mae_global']}, RMSE={seasonal_test_summary['rmse_global']}")

    _ = (
        json,
        joblib,
        optuna,
        EXOGENOUS_COLS,
        CALENDAR_COLS,
        make_model,
        train_model,
        suggest_xgboost_params,
        evaluate_multistep,
        seasonal_naive_predictions_from_windows,
        save_metrics,
        optimize_xgboost,
        temporal_split,
        print_split_summary,
        format_split_shapes,
    )


if __name__ == "__main__":
    main()
