from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = Path("data/processed/spot_es_with_exogenous.parquet")
RESULTS_DIR = Path("results/chronos")
FINETUNED_OOF_PATH = Path("models/chronos2_with_covariates/models/Chronos2FineTuned/utils/oof.pkl")
ZEROSHOT_OOF_PATH = Path("models/chronos2_with_covariates/models/Chronos2ZeroShot/utils/oof.pkl")


def normalize_timestamp_column(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column])
    try:
        df[column] = df[column].dt.tz_localize(None)
    except TypeError:
        pass
    return df


def load_oof_predictions(path: Path) -> pd.DataFrame:
    obj = pd.read_pickle(path)

    if not isinstance(obj, list):
        raise TypeError(f"{path} does not contain a list")
    if len(obj) == 0:
        raise ValueError(f"{path} contains an empty list")

    first = obj[0]
    if isinstance(first, pd.DataFrame):
        df = first.copy()
    else:
        df = pd.DataFrame(first)

    df = df.reset_index()

    if "timestamp" not in df.columns:
        raise KeyError(f"{path} is missing 'timestamp' column after reset_index")
    if "mean" not in df.columns:
        raise KeyError(f"{path} is missing 'mean' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[["timestamp", "mean"]].rename(columns={"mean": "y_pred"})
    return normalize_timestamp_column(df)


def load_true_values() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[["timestamp", "price"]].rename(columns={"price": "y_true"})
    return normalize_timestamp_column(df)


def evaluate_predictions(name: str, pred_df: pd.DataFrame, true_df: pd.DataFrame) -> dict:
    pred_df = normalize_timestamp_column(pred_df)
    true_df = normalize_timestamp_column(true_df)
    merged = pred_df.merge(true_df, on="timestamp", how="left")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    if len(merged) != len(pred_df):
        raise ValueError(f"{name}: merged rows {len(merged)} != prediction rows {len(pred_df)}")
    if merged["y_true"].isna().any():
        missing = int(merged["y_true"].isna().sum())
        raise ValueError(f"{name}: {missing} predictions could not be aligned with true values")

    mae = float(mean_absolute_error(merged["y_true"], merged["y_pred"]))
    rmse = float(np.sqrt(mean_squared_error(merged["y_true"], merged["y_pred"])))

    output_path = RESULTS_DIR / f"{name}_oof_predictions_aligned.csv"
    merged.to_csv(output_path, index=False)

    return {
        "model": name,
        "n_points": int(len(merged)),
        "start_timestamp": merged["timestamp"].min().isoformat(),
        "end_timestamp": merged["timestamp"].max().isoformat(),
        "mae": mae,
        "rmse": rmse,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    true_df = load_true_values()
    finetuned_pred_df = load_oof_predictions(FINETUNED_OOF_PATH)
    zeroshot_pred_df = load_oof_predictions(ZEROSHOT_OOF_PATH)

    finetuned_result = evaluate_predictions("Chronos2FineTuned", finetuned_pred_df, true_df)
    zeroshot_result = evaluate_predictions("Chronos2ZeroShot", zeroshot_pred_df, true_df)

    results = [finetuned_result, zeroshot_result]

    output_json = RESULTS_DIR / "chronos_oof_metrics.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Chronos2FineTuned: MAE={finetuned_result['mae']}, RMSE={finetuned_result['rmse']}")
    print(f"Chronos2ZeroShot: MAE={zeroshot_result['mae']}, RMSE={zeroshot_result['rmse']}")


if __name__ == "__main__":
    main()
