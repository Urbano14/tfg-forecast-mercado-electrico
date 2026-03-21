from datetime import datetime

from pydantic import BaseModel


class ForecastPointResponse(BaseModel):
    timestamp: datetime
    value: float


class ForecastResponse(BaseModel):
    model: str
    requested_date: datetime
    horizon_hours: int
    forecast: list[ForecastPointResponse]