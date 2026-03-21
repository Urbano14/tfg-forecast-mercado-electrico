from datetime import datetime, timedelta

from fastapi import HTTPException

from app.application.services.model_service import is_supported_model
from app.schemas.forecast import ForecastPointResponse, ForecastResponse


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