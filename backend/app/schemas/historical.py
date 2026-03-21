from datetime import datetime

from pydantic import BaseModel


class HistoricalDataResponse(BaseModel):
    timestamp: datetime
    price: float
    demand_forecast: float | None = None
    wind_forecast: float | None = None
    solar_forecast: float | None = None
    hydro_programmed: float | None = None