from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.evaluation.split import load_data, temporal_split
from src.evaluation.backtesting import rolling_origin_backtest
from src.modelos.naive import NaiveModel
from src.modelos.seasonal_naive import SeasonalNaiveModel


RESULTS_PATH = Path("results/baselines_results.csv")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_one(name: str, model, series, dataset_name: str, horizon=24, stride=24):
    res = rolling_origin_backtest(
        series=series,
        model=model,
        horizon=horizon,
        stride=stride
    )

    print(
        f"{dataset_name} | {name}: "
        f"n_origins={res.n_origins} | "
        f"MAE={res.mae:.4f} | "
        f"RMSE={res.rmse:.4f}"
    )

    return {
        "dataset": dataset_name,
        "model": name,
        "n_origins": res.n_origins,
        "horizon": res.horizon,
        "stride": res.stride,
        "mae": res.mae,
        "rmse": res.rmse,
    }


def main():
    df = load_data()
    train, val, test = temporal_split(df)

    results = []

    print("=== Running Baselines ===")

    
    results.append(run_one("Naive", NaiveModel(), val["price"], "validation"))
    results.append(run_one("SeasonalNaive(24)", SeasonalNaiveModel(24), val["price"], "validation"))

    
    results.append(run_one("Naive", NaiveModel(), test["price"], "test"))
    results.append(run_one("SeasonalNaive(24)", SeasonalNaiveModel(24), test["price"], "test"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()