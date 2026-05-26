# Centraliza la lista de modelos que el backend reconoce.

AVAILABLE_MODELS = [
    {
        "id": "seasonal_naive",
        "name": "Seasonal Naive",
        "type": "baseline",
        "horizon_hours": 24
    },
    {
        "id": "xgboost",
        "name": "XGBoost",
        "type": "machine_learning",
        "horizon_hours": 24
    },
    {
        "id": "chronos",
        "name": "Chronos-2 Fine-Tuned",
        "type": "foundation_model",
        "horizon_hours": 24
    }
]


def get_available_models():
    return AVAILABLE_MODELS


def is_supported_model(model_id: str) -> bool:
    return any(model["id"] == model_id for model in AVAILABLE_MODELS)