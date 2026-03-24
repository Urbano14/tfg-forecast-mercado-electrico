from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.application.services.historical_service import (
    get_historical_data_between,
    get_historical_data_range,
    get_previous_24_hours,
)
from app.application.services.model_service import is_supported_model
from app.schemas.forecast import ForecastPointResponse, ForecastResponse
from app.infrastructure.ml.xgboost_loader import load_xgboost_model


def generate_dummy_forecast(
    requested_date: datetime,
    model: str
) -> ForecastResponse:
    if not is_supported_model(model):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not supported"
        )

    forecast = [
        ForecastPointResponse(
            timestamp=requested_date + timedelta(hours=i + 1),
            value=0.0
        )
        for i in range(24)
    ]

    return ForecastResponse(
        model=model,
        model_type="unknown",
        requested_date=requested_date,
        horizon_hours=24,
        forecast=forecast
    )


def generate_seasonal_naive_forecast(
    db: Session,
    requested_date: datetime
) -> ForecastResponse:

    data_range = get_historical_data_range(db)
    if data_range["start"] is None or data_range["end"] is None:
        raise HTTPException(
            status_code=400,
            detail="No historical data available"
        )

    if requested_date.minute != 0 or requested_date.second != 0:
        raise HTTPException(
            status_code=400,
            detail="Date must be aligned to full hour (e.g., 2022-01-01T00:00:00)"
        )

    if requested_date <= data_range["start"]:
        raise HTTPException(
            status_code=400,
            detail="Requested date is too early"
        )

    if requested_date > data_range["end"]:
        raise HTTPException(
            status_code=400,
            detail="Requested date is beyond available data"
        )

    previous_24h = get_previous_24_hours(db=db, requested_date=requested_date)

    if len(previous_24h) != 24:
        raise HTTPException(
            status_code=400,
            detail="Not enough historical data: need previous 24 hours"
        )

    forecast = [
        ForecastPointResponse(
            timestamp=requested_date + timedelta(hours=i + 1),
            value=previous_24h[i].price
        )
        for i in range(24)
    ]

    return ForecastResponse(
    model="seasonal_naive",
    model_type="baseline",
    requested_date=requested_date,
    horizon_hours=24,
    forecast=forecast
    )

XGBOOST_FEATURE_COLS = [
    "lag_1",
    "lag_24",
    "lag_168",
    "demand_forecast",
    "wind_forecast",
    "solar_forecast",
    "hydro_programmed",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


def _build_xgboost_features_for_timestamp(
    ts: datetime,
    price_by_ts: dict[datetime, float],
    exog_by_ts: dict[datetime, dict[str, float]],
) -> pd.DataFrame:
    lag_1_ts = ts - timedelta(hours=1)
    lag_24_ts = ts - timedelta(hours=24)
    lag_168_ts = ts - timedelta(hours=168)

    if lag_1_ts not in price_by_ts:
        raise HTTPException(status_code=400, detail="Missing lag_1 data for XGBoost")
    if lag_24_ts not in price_by_ts:
        raise HTTPException(status_code=400, detail="Missing lag_24 data for XGBoost")
    if lag_168_ts not in price_by_ts:
        raise HTTPException(status_code=400, detail="Missing lag_168 data for XGBoost")
    if ts not in exog_by_ts:
        raise HTTPException(status_code=400, detail="Missing exogenous data for XGBoost")

    hour = ts.hour
    dayofweek = ts.weekday()
    month = ts.month
    is_weekend = 1 if dayofweek >= 5 else 0

    row = {
        "lag_1": price_by_ts[lag_1_ts],
        "lag_24": price_by_ts[lag_24_ts],
        "lag_168": price_by_ts[lag_168_ts],
        "demand_forecast": exog_by_ts[ts]["demand_forecast"],
        "wind_forecast": exog_by_ts[ts]["wind_forecast"],
        "solar_forecast": exog_by_ts[ts]["solar_forecast"],
        "hydro_programmed": exog_by_ts[ts]["hydro_programmed"],
        "is_weekend": is_weekend,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow_sin": float(np.sin(2 * np.pi * dayofweek / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dayofweek / 7)),
        "month_sin": float(np.sin(2 * np.pi * (month - 1) / 12)),
        "month_cos": float(np.cos(2 * np.pi * (month - 1) / 12)),
    }

    return pd.DataFrame([row], columns=XGBOOST_FEATURE_COLS)

def generate_xgboost_forecast(
    db: Session,
    requested_date: datetime
) -> ForecastResponse:
    if requested_date.minute != 0 or requested_date.second != 0:
        raise HTTPException(
            status_code=400,
            detail="Date must be aligned to full hour (e.g., 2022-01-01T00:00:00)"
        )

    start = requested_date - timedelta(hours=168)
    end = requested_date + timedelta(hours=24)
    rows = get_historical_data_between(db, start, end)

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No historical data available for XGBoost"
        )

    price_by_ts: dict[datetime, float] = {}
    exog_by_ts: dict[datetime, dict[str, float]] = {}

    for row in rows:
        ts = row.timestamp
        exog_by_ts[ts] = {
            "demand_forecast": row.demand_forecast,
            "wind_forecast": row.wind_forecast,
            "solar_forecast": row.solar_forecast,
            "hydro_programmed": row.hydro_programmed,
        }
        if ts <= requested_date:
            price_by_ts[ts] = row.price

    if requested_date not in price_by_ts:
        raise HTTPException(
            status_code=400,
            detail="Requested date not present in historical data for XGBoost"
        )

    model = load_xgboost_model()

    forecast = []
    for i in range(24):
        ts = requested_date + timedelta(hours=i + 1)
        X = _build_xgboost_features_for_timestamp(ts, price_by_ts, exog_by_ts)
        pred = float(model.predict(X)[0])
        price_by_ts[ts] = pred

        forecast.append(
            ForecastPointResponse(
                timestamp=ts,
                value=pred
            )
        )

    return ForecastResponse(
    model="xgboost",
    model_type="machine_learning",
    requested_date=requested_date,
    horizon_hours=24,
    forecast=forecast
    )
