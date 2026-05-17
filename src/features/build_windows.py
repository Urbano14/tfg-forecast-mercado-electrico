from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DATA_PATH = Path("data/processed/spot_es_with_exogenous.parquet")
DEFAULT_INPUT_WINDOW = 168
DEFAULT_HORIZON = 24
TARGET_COL = "price"

EXOGENOUS_COLS: Sequence[str] = [
    "demand_forecast",
    "wind_forecast",
    "solar_forecast",
    "hydro_programmed",
]

CALENDAR_COLS: Sequence[str] = [
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    hour = df["timestamp"].dt.hour
    dayofweek = df["timestamp"].dt.dayofweek
    month = df["timestamp"].dt.month

    df["is_weekend"] = (dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def load_series(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_calendar_features(df)
    return df


def build_supervised_windows(
    df: pd.DataFrame,
    input_window: int = DEFAULT_INPUT_WINDOW,
    horizon: int = DEFAULT_HORIZON,
    target_col: str = TARGET_COL,
    past_covariate_cols: Sequence[str] | None = None,
    future_covariate_cols: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
   
    required_cols = {"timestamp", target_col}
    missing_required = required_cols.difference(df.columns)
    if missing_required:
        missing_str = ", ".join(sorted(missing_required))
        raise ValueError(f"Missing required columns: {missing_str}")

    past_covariate_cols = [] if past_covariate_cols is None else list(past_covariate_cols)
    future_covariate_cols = [] if future_covariate_cols is None else list(future_covariate_cols)

    requested_cols = set(past_covariate_cols).union(future_covariate_cols)
    missing_covariates = requested_cols.difference(df.columns)
    if missing_covariates:
        missing_str = ", ".join(sorted(missing_covariates))
        raise ValueError(f"Missing covariate columns: {missing_str}")

    df = df.sort_values("timestamp").reset_index(drop=True)

    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    timestamps: list[pd.Timestamp] = []

    for i in range(input_window, len(df) - horizon + 1):
        past_window = df.iloc[i - input_window : i]
        future_window = df.iloc[i : i + horizon]

        used_past_cols = [target_col, *past_covariate_cols]
        if past_window[used_past_cols].isna().to_numpy().any():
            continue

        if future_window[[target_col, *future_covariate_cols]].isna().to_numpy().any():
            continue

        X_parts = [past_window[target_col].to_numpy(dtype=float)]

        for col in past_covariate_cols:
            X_parts.append(past_window[col].to_numpy(dtype=float))

        for col in future_covariate_cols:
            X_parts.append(future_window[col].to_numpy(dtype=float))

        X_rows.append(np.concatenate(X_parts))
        y_rows.append(future_window[target_col].to_numpy(dtype=float))
        timestamps.append(pd.to_datetime(df.loc[i, "timestamp"]))

    if not X_rows:
        return (
            np.empty((0, 0), dtype=float),
            np.empty((0, horizon), dtype=float),
            pd.DatetimeIndex([]),
        )

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    timestamp_index = pd.DatetimeIndex(timestamps)
    return X, y, timestamp_index
