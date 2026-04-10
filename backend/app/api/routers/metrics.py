from fastapi import APIRouter

from app.application.services.metrics_service import get_model_metrics
from app.schemas.metrics import MetricsResponse

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", response_model=MetricsResponse)
@router.get("/", response_model=MetricsResponse)
def get_metrics():
    return {"metrics": get_model_metrics()}
