from datetime import datetime

from pydantic import BaseModel

# Estos schemas definen la estructura de los datos que se devolverán en la respuesta de la API de forecast. 


# El ForecastPointResponse representa un punto individual en el forecast, con su timestamp y valor. 
class ForecastPointResponse(BaseModel):
    timestamp: datetime
    value: float

#  representa la respuesta completa del forecast, incluyendo información sobre el modelo utilizado, 
# la fecha solicitada, el horizonte de horas y una lista de puntos de forecast.
class ForecastResponse(BaseModel):
    model: str
    model_type: str
    requested_date: datetime
    horizon_hours: int
    forecast: list[ForecastPointResponse]