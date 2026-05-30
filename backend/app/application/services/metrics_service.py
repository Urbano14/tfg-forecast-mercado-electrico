from app.application.services.model_service import get_available_models

# Devuelve métricas ya fijadas a partir de los resultados experimentales.

METRICS_BY_MODEL = {
    "seasonal_naive": {
        "mae": 19.0937,
        "rmse": 28.0507,
    },
    "xgboost": {
        "mae": 17.0440,
        "rmse": 22.4152,
    },
    "chronos": {
        "mae": 10.6739,
        "rmse": 15.6158,
    },
}

# Devuelve las métricas de cada modelo disponible, si es que se han fijado.
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
