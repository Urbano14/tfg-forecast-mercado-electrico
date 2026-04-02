from pydantic import BaseModel


class ModelMetricResponse(BaseModel):
    id: str
    name: str
    type: str
    mae: float
    rmse: float


class MetricsResponse(BaseModel):
    metrics: list[ModelMetricResponse]
