from fastapi import APIRouter

from app.application.services.model_service import get_available_models

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/")
def get_models():
    return {
        "models": get_available_models()
    }