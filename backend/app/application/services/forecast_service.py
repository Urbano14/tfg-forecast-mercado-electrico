from datetime import datetime, timedelta

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.application.services.historical_service import get_previous_24_hours
from app.application.services.model_service import is_supported_model
from app.schemas.forecast import ForecastPointResponse, ForecastResponse
from app.application.services.historical_service import get_historical_data_range


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
        requested_date=requested_date,
        horizon_hours=24,
        forecast=forecast
    )


def generate_seasonal_naive_forecast(
    db: Session,
    requested_date: datetime
) -> ForecastResponse:

    data_range = get_historical_data_range(db)
    print("requested_date:", requested_date)
    print("range_start:", data_range["start"])
    print("range_end:", data_range["end"])

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

    print("previous_24h len:", len(previous_24h))

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
        requested_date=requested_date,
        horizon_hours=24,
        forecast=forecast
    )