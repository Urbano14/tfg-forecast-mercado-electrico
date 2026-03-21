import pandas as pd

from app.core.database import SessionLocal
from app.infrastructure.db.models import MarketData


def main():
    df = pd.read_parquet("data/processed/spot_es_with_exogenous.parquet")

    df = df[
        [
            "timestamp",
            "price",
            "demand_forecast",
            "wind_forecast",
            "solar_forecast",
            "hydro_programmed",
        ]
    ].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    records = df.to_dict(orient="records")

    db = SessionLocal()
    try:
        objects = [MarketData(**record) for record in records]
        db.bulk_save_objects(objects)
        db.commit()
        print(f"Inserted {len(objects)} rows into market_data")
    finally:
        db.close()


if __name__ == "__main__":
    main()