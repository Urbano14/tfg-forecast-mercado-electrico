from fastapi import APIRouter

from app.application.services.model_service import get_available_models
from app.schemas.models import ModelsListResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/", response_model=ModelsListResponse)
def get_models():
    return {
        "models": get_available_models()
    }