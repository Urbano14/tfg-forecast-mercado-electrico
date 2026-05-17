from app.application.services.model_service import get_available_models

METRICS_BY_MODEL = {
    "seasonal_naive": {
        "mae": 18.2531,
        "rmse": 27.6373,
    },
    "xgboost": {
        "mae": 14.3392,
        "rmse": 18.9978,
    },
    "chronos": {
        "mae": 11.6566,
        "rmse": 14.4666,
    },
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
