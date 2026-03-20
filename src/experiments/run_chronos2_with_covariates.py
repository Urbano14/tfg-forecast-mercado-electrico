from __future__ import annotations

from pathlib import Path
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


DATA_PATH = Path("data/processed/spot_es_with_exogenous.parquet")
RESULTS_PATH = Path("results/chronos2_with_covariates_results.csv")
MODEL_DIR = Path("models/chronos2_with_covariates")


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def prepare_data():
    df = load_data()

    covariates = [
        "demand_forecast",
        "wind_forecast",
        "solar_forecast",
        "hydro_programmed",
    ]

    df = df[["timestamp", "price"] + covariates].copy()

    df = df.rename(columns={"price": "target"})
    df["item_id"] = "precio_es"

    df = df[["item_id", "timestamp", "target"] + covariates]

    train_df = df[df["timestamp"].dt.year <= 2023].copy()
    test_df = df[df["timestamp"].dt.year == 2024].copy()

    train_ts = TimeSeriesDataFrame.from_data_frame(
        train_df,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    test_ts = TimeSeriesDataFrame.from_data_frame(
        test_df,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    return train_ts, test_ts, covariates


def main():
    print("=== Chronos-2 with covariates ===")

    prediction_length = 24
    train_ts, test_ts, covariates = prepare_data()

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target="target",
        known_covariates_names=covariates,
        eval_metric="MAE",
        path=str(MODEL_DIR),
    )

    predictor.fit(
        train_data=train_ts,
        hyperparameters={
            "Chronos2": [
                # Zero-shot
                {"ag_args": {"name_suffix": "ZeroShot"}},
                # Fine-tuned
                {
                    "fine_tune": True,
                    "ag_args": {"name_suffix": "FineTuned"},
                },
            ]
        },
        enable_ensemble=False,
        verbosity=2,
        time_limit=1800,  
    )

    leaderboard = predictor.leaderboard(test_ts)
    print("\n=== Leaderboard ===")
    print(leaderboard)

    leaderboard.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()