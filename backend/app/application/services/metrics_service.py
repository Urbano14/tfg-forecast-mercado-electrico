from app.application.services.model_service import get_available_models

METRICS_BY_MODEL = {
    "seasonal_naive": {"mae": 18.0948, "rmse": 25.0},
    "xgboost": {"mae": 6.4775, "rmse": 9.8},
    "chronos": {"mae": 4.1165, "rmse": 6.5},
}


def get_model_metrics():
    metrics = []

    for model in get_available_models():
        model_id = model["id"]
        if model_id not in METRICS_BY_MODEL:
            continue

        model_metrics = METRICS_BY_MODEL[model_id]
        metrics.append(
            {
                "id": model_id,
                "name": model["name"],
                "type": model["type"],
                "mae": model_metrics["mae"],
                "rmse": model_metrics["rmse"],
            }
        )

    return metrics
