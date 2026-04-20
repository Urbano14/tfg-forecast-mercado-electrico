from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.application.services.forecast_service import (
    generate_dummy_forecast,
    generate_chronos_forecast,
    generate_seasonal_naive_forecast,
    generate_xgboost_forecast
)
from app.core.database import get_db
from app.schemas.forecast import ForecastResponse

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("", response_model=ForecastResponse)
@router.get("/", response_model=ForecastResponse)
def get_forecast(
    date: datetime,
    model: str,
    db: Session = Depends(get_db)
):
    if model == "seasonal_naive":
        return generate_seasonal_naive_forecast(db, date)

    if model == "xgboost":
        return generate_xgboost_forecast(db, date)

    if model == "chronos":
        return generate_chronos_forecast(db, date)

    return generate_dummy_forecast(
        requested_date=date,
        model=model
    )
