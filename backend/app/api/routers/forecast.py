from datetime import datetime

from fastapi import APIRouter

from app.application.services.forecast_service import generate_dummy_forecast
from app.schemas.forecast import ForecastResponse

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/", response_model=ForecastResponse)
def get_forecast(
    date: datetime,
    model: str
):
    return generate_dummy_forecast(
        requested_date=date,
        model=model
    )