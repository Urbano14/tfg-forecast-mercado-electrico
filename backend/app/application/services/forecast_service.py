from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from fastapi import HTTPException
from sqlalchemy.orm import Session
from autogluon.timeseries import TimeSeriesDataFrame

# Servicios de histórico: se usan para consultar PostgreSQL desde la lógica de forecast.
from app.application.services.historical_service import (
    get_historical_data_between,
    get_historical_data_range,
    get_previous_24_hours,
)
from app.application.services.model_service import is_supported_model
from app.schemas.forecast import ForecastPointResponse, ForecastResponse

# Loaders de modelos: el backend no entrena modelos, solo carga artefactos ya generados.
from app.infrastructure.ml.xgboost_loader import (
    load_xgboost_model,
    load_xgboost_multistep_complete_model,
    load_xgboost_multistep_minimal_model,
)
from app.infrastructure.ml.chronos_loader import load_chronos_predictor


# Forecast falso de prueba.
def generate_dummy_forecast(
    requested_date: datetime,
    model: str
) -> ForecastResponse:
    # Comprueba que el nombre del modelo esté registrado como soportado.
    if not is_supported_model(model):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not supported"
        )

    # Genera 24 valores futuros con precio 0.0, uno por cada hora.
    forecast = [
        ForecastPointResponse(
            timestamp=requested_date + timedelta(hours=i + 1),
            value=0.0
        )
        for i in range(24)
    ]

    return ForecastResponse(
        model=model,
        model_type="unknown",
        requested_date=requested_date,
        horizon_hours=24,
        forecast=forecast
    )


# Genera una predicción Seasonal Naive: copia las 24 horas anteriores como próximas 24 horas.
def generate_seasonal_naive_forecast(
    db: Session,
    requested_date: datetime
) -> ForecastResponse:

    # Normaliza la fecha quitando zona horaria si viene con ella, para compararla con PostgreSQL.
    if requested_date.tzinfo is not None:
        requested_date = requested_date.replace(tzinfo=None)

    # Consulta el rango disponible en la tabla market_data.
    data_range = get_historical_data_range(db)
    if data_range["start"] is None or data_range["end"] is None:
        raise HTTPException(
            status_code=400,
            detail="No historical data available"
        )

    # Normaliza también el rango por si la base de datos devuelve timestamps con zona horaria.
    if data_range["start"].tzinfo is not None:
        data_range["start"] = data_range["start"].replace(tzinfo=None)
    if data_range["end"].tzinfo is not None:
        data_range["end"] = data_range["end"].replace(tzinfo=None)

    # El forecast trabaja con datos horarios, así que la fecha base debe estar alineada a una hora exacta.
    if requested_date.minute != 0 or requested_date.second != 0:
        raise HTTPException(
            status_code=400,
            detail="Date must be aligned to full hour (e.g., 2022-01-01T00:00:00)"
        )

    # No se puede predecir si la fecha está al inicio del dataset, porque faltan 24 horas anteriores.
    if requested_date <= data_range["start"]:
        raise HTTPException(
            status_code=400,
            detail="Requested date is too early"
        )

    # No se permite pedir una fecha base posterior al último dato disponible.
    if requested_date > data_range["end"]:
        raise HTTPException(
            status_code=400,
            detail="Requested date is beyond available data"
        )

    # Recupera exactamente las 24 horas anteriores a requested_date.
    previous_24h = get_previous_24_hours(db=db, requested_date=requested_date)

    # Seasonal Naive necesita las 24 horas previas completas.
    if len(previous_24h) != 24:
        raise HTTPException(
            status_code=400,
            detail="Not enough historical data: need previous 24 hours"
        )

    # Copia las 24 horas anteriores como predicción de las siguientes 24 horas.
    forecast = [
        ForecastPointResponse(
            timestamp=requested_date + timedelta(hours=i + 1),
            value=previous_24h[i].price
        )
        for i in range(24)
    ]

    return ForecastResponse(
        model="seasonal_naive",
        model_type="baseline",
        requested_date=requested_date,
        horizon_hours=24,
        forecast=forecast
    )


# Columnas del enfoque XGBoost one-step antiguo. Se conserva por compatibilidad, pero el flujo final usa multi-step.
XGBOOST_FEATURE_COLS = [
    "lag_1",
    "lag_24",
    "lag_168",
    "demand_forecast",
    "wind_forecast",
    "solar_forecast",
    "hydro_programmed",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

# Configuración del XGBoost final multi-step: 168 horas pasadas para predecir 24 horas futuras.
XGBOOST_INPUT_WINDOW = 168
XGBOOST_HORIZON = 24


# Construye la ventana histórica de 168 precios que necesita XGBoost multi-step.
def _build_price_window_168(
    requested_date: datetime,
    price_by_ts: dict[datetime, float],
) -> np.ndarray:
    # Crea los timestamps desde requested_date-167h hasta requested_date, en orden cronológico.
    timestamps = [
        requested_date - timedelta(hours=offset)
        for offset in range(XGBOOST_INPUT_WINDOW - 1, -1, -1)
    ]

    # Comprueba que existan todos los precios necesarios para la ventana de 168 horas.
    if any(ts not in price_by_ts for ts in timestamps):
        raise HTTPException(
            status_code=400,
            detail="Not enough historical price data for XGBoost multi-step (need 168 hours)"
        )

    # Convierte los precios históricos a array NumPy en el mismo orden temporal de entrenamiento.
    price_window = np.asarray([price_by_ts[ts] for ts in timestamps], dtype=float)

    # Validación defensiva: la ventana debe tener exactamente 168 valores.
    if price_window.shape[0] != XGBOOST_INPUT_WINDOW:
        raise HTTPException(
            status_code=400,
            detail="Invalid historical price window for XGBoost multi-step"
        )

    return price_window


# Genera los timestamps futuros que se van a predecir.
def _future_timestamps(
    requested_date: datetime,
    horizon: int = XGBOOST_HORIZON,
) -> list[datetime]:
    # Si requested_date es la última hora observada, la predicción empieza en requested_date + 1h.
    return [requested_date + timedelta(hours=i) for i in range(1, horizon + 1)]


# Calcula las variables calendario futuras para las 24 horas que se van a predecir.
def _calendar_features_for_timestamps(
    timestamps: list[datetime],
) -> np.ndarray:
    rows = []
    for ts in timestamps:
        hour = ts.hour
        dayofweek = ts.weekday()
        month = ts.month
        is_weekend = 1 if dayofweek >= 5 else 0

        # Se codifican hora, día de semana y mes con seno/coseno para representar ciclos temporales.
        rows.extend(
            [
                float(is_weekend),
                float(np.sin(2 * np.pi * hour / 24)),
                float(np.cos(2 * np.pi * hour / 24)),
                float(np.sin(2 * np.pi * dayofweek / 7)),
                float(np.cos(2 * np.pi * dayofweek / 7)),
                float(np.sin(2 * np.pi * month / 12)),
                float(np.cos(2 * np.pi * month / 12)),
            ]
        )

    # Devuelve 7 variables calendario * 24 horas = 168 valores.
    return np.asarray(rows, dtype=float)


# Intenta construir las exógenas futuras para las próximas 24 horas.
def _future_exogenous_features(
    timestamps: list[datetime],
    exog_by_ts: dict[datetime, dict[str, float]],
) -> np.ndarray | None:
    values = []
    for ts in timestamps:
        # Busca las exógenas correspondientes a esa hora futura.
        exog = exog_by_ts.get(ts)
        if exog is None:
            return None

        current_values = [
            exog.get("demand_forecast"),
            exog.get("wind_forecast"),
            exog.get("solar_forecast"),
            exog.get("hydro_programmed"),
        ]

        # Si falta alguna exógena, no se puede usar el modelo completo.
        if any(value is None for value in current_values):
            return None

        values.extend(float(value) for value in current_values)

    # Devuelve 4 variables exógenas * 24 horas = 96 valores.
    return np.asarray(values, dtype=float)


# Construye la entrada final para XGBoost multi-step y decide si usar variante mínima o completa.
def _build_xgboost_multistep_input(
    requested_date: datetime,
    price_by_ts: dict[datetime, float],
    exog_by_ts: dict[datetime, dict[str, float]],
) -> tuple[np.ndarray, str]:
    # Parte 1: 168 precios pasados.
    price_window = _build_price_window_168(requested_date, price_by_ts)

    # Parte 2: timestamps de las 24 horas futuras.
    future_ts = _future_timestamps(requested_date)

    # Parte 3: calendario futuro, siempre disponible porque depende de la fecha.
    calendar_features = _calendar_features_for_timestamps(future_ts)

    # Parte 4: exógenas futuras. Pueden faltar, por eso puede devolver None.
    exogenous_features = _future_exogenous_features(future_ts, exog_by_ts)

    if exogenous_features is not None:
        # Variante completa: 168 precios + 168 calendario + 96 exógenas = 432 columnas.
        X = np.concatenate([price_window, calendar_features, exogenous_features])
        variant = "complete"
    else:
        # Variante mínima: 168 precios + 168 calendario = 336 columnas.
        X = np.concatenate([price_window, calendar_features])
        variant = "minimal"

    # El modelo espera una matriz 2D, aunque solo haya una muestra: shape (1, n_features).
    return X.reshape(1, -1), variant


# Construye una fila XGBoost one-step con lags. Pertenece al enfoque anterior, no al flujo multi-step final.
def _build_xgboost_features_for_timestamp(
    ts: datetime,
    price_by_ts: dict[datetime, float],
    exog_by_ts: dict[datetime, dict[str, float]],
) -> pd.DataFrame:
    lag_1_ts = ts - timedelta(hours=1)
    lag_24_ts = ts - timedelta(hours=24)
    lag_168_ts = ts - timedelta(hours=168)

    # Valida que estén disponibles los lags necesarios.
    if lag_1_ts not in price_by_ts:
        raise HTTPException(status_code=400, detail="Missing lag_1 data for XGBoost")
    if lag_24_ts not in price_by_ts:
        raise HTTPException(status_code=400, detail="Missing lag_24 data for XGBoost")
    if lag_168_ts not in price_by_ts:
        raise HTTPException(status_code=400, detail="Missing lag_168 data for XGBoost")
    if ts not in exog_by_ts:
        raise HTTPException(status_code=400, detail="Missing exogenous data for XGBoost")

    hour = ts.hour
    dayofweek = ts.weekday()
    month = ts.month
    is_weekend = 1 if dayofweek >= 5 else 0

    # Fila de features del enfoque one-step: lags + exógenas + calendario.
    row = {
        "lag_1": price_by_ts[lag_1_ts],
        "lag_24": price_by_ts[lag_24_ts],
        "lag_168": price_by_ts[lag_168_ts],
        "demand_forecast": exog_by_ts[ts]["demand_forecast"],
        "wind_forecast": exog_by_ts[ts]["wind_forecast"],
        "solar_forecast": exog_by_ts[ts]["solar_forecast"],
        "hydro_programmed": exog_by_ts[ts]["hydro_programmed"],
        "is_weekend": is_weekend,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow_sin": float(np.sin(2 * np.pi * dayofweek / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dayofweek / 7)),
        "month_sin": float(np.sin(2 * np.pi * (month - 1) / 12)),
        "month_cos": float(np.cos(2 * np.pi * (month - 1) / 12)),
    }

    return pd.DataFrame([row], columns=XGBOOST_FEATURE_COLS)


# Genera el forecast XGBoost multi-step usado por la aplicación.
def generate_xgboost_forecast(
    db: Session,
    requested_date: datetime
) -> ForecastResponse:
    # Normaliza la fecha base.
    if requested_date.tzinfo is not None:
        requested_date = requested_date.replace(tzinfo=None)

    # La fecha debe estar alineada a una hora exacta.
    if requested_date.minute != 0 or requested_date.second != 0:
        raise HTTPException(
            status_code=400,
            detail="Date must be aligned to full hour (e.g., 2022-01-01T00:00:00)"
        )

    # XGBoost necesita 168 horas hasta requested_date y 24 horas futuras para posibles exógenas.
    start = requested_date - timedelta(hours=XGBOOST_INPUT_WINDOW - 1)
    end = requested_date + timedelta(hours=XGBOOST_HORIZON)
    rows = get_historical_data_between(db, start, end)

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No historical data available for XGBoost"
        )

    # Diccionario de precios históricos y diccionario de exógenas por timestamp.
    price_by_ts: dict[datetime, float] = {}
    exog_by_ts: dict[datetime, dict[str, float]] = {}

    for row in rows:
        ts = row.timestamp
        exog_by_ts[ts] = {
            "demand_forecast": row.demand_forecast,
            "wind_forecast": row.wind_forecast,
            "solar_forecast": row.solar_forecast,
            "hydro_programmed": row.hydro_programmed,
        }
        # Para los precios solo se usan horas observadas hasta requested_date.
        if ts <= requested_date:
            price_by_ts[ts] = row.price

    # La fecha base debe existir en histórico porque actúa como última hora observada.
    if requested_date not in price_by_ts:
        raise HTTPException(
            status_code=400,
            detail="Requested date not present in historical data for XGBoost"
        )

    # Construye la entrada en el mismo formato que el entrenamiento y decide minimal/complete.
    X, variant = _build_xgboost_multistep_input(requested_date, price_by_ts, exog_by_ts)

    # Si hay exógenas futuras completas usa el modelo completo; si no, usa el mínimo.
    if variant == "complete":
        model = load_xgboost_multistep_complete_model()
    else:
        model = load_xgboost_multistep_minimal_model()

    # El modelo devuelve un vector de 24 predicciones.
    pred = model.predict(X)[0]
    if len(pred) != XGBOOST_HORIZON:
        raise HTTPException(
            status_code=500,
            detail=f"XGBoost multi-step model returned {len(pred)} steps, expected {XGBOOST_HORIZON}"
        )

    # Asocia cada valor predicho a su timestamp futuro.
    future_ts = _future_timestamps(requested_date)
    forecast = [
        ForecastPointResponse(
            timestamp=ts,
            value=float(value)
        )
        for ts, value in zip(future_ts, pred)
    ]

    return ForecastResponse(
        model="xgboost",
        model_type="machine_learning",
        requested_date=requested_date,
        horizon_hours=XGBOOST_HORIZON,
        forecast=forecast
    )


# Genera XGBoost tomando como fecha base la última hora disponible en la base de datos.
#Serviría para predecir directamente "mañana" (No se usa).
def generate_xgboost_forecast_from_latest_available(
    db: Session,
) -> ForecastResponse:
    data_range = get_historical_data_range(db)
    if data_range["end"] is None:
        raise HTTPException(
            status_code=400,
            detail="No historical data available for XGBoost"
        )

    latest_date = data_range["end"]
    if latest_date.tzinfo is not None:
        latest_date = latest_date.replace(tzinfo=None)

    return generate_xgboost_forecast(db, latest_date)


# Genera el forecast con Chronos/AutoGluon.
def generate_chronos_forecast(
    db: Session,
    requested_date: datetime
) -> ForecastResponse:
    # Normaliza la fecha base.
    if requested_date.tzinfo is not None:
        requested_date = requested_date.replace(tzinfo=None)

    # Chronos también trabaja con fecha base alineada a hora completa.
    if requested_date.minute != 0 or requested_date.second != 0:
        raise HTTPException(
            status_code=400,
            detail="Date must be aligned to full hour (e.g., 2022-01-01T00:00:00)"
        )

    # Consulta el rango disponible en base de datos.
    data_range = get_historical_data_range(db)
    if data_range["start"] is None or data_range["end"] is None:
        raise HTTPException(
            status_code=400,
            detail="No historical data available for Chronos"
        )

    if requested_date < data_range["start"]:
        raise HTTPException(
            status_code=400,
            detail="Requested date is too early for Chronos"
        )

    # Chronos usa todo el histórico disponible hasta requested_date, no solo 168 horas.
    historical_rows = get_historical_data_between(
        db=db,
        start=data_range["start"],
        end=requested_date
    )

    if not historical_rows:
        raise HTTPException(
            status_code=400,
            detail="No historical data available for Chronos"
        )

    # Deduplica por timestamp por si el dataset tuviera registros horarios duplicados.
    hist_by_ts: dict[datetime, any] = {}
    for row in historical_rows:
        if row.timestamp not in hist_by_ts:
            hist_by_ts[row.timestamp] = row
    historical_rows = list(hist_by_ts.values())

    # La última fila histórica debe llegar exactamente hasta requested_date.
    if historical_rows[-1].timestamp < requested_date:
        raise HTTPException(
            status_code=400,
            detail="Requested date not present in historical data for Chronos"
        )

    # Prepara el histórico en formato AutoGluon: item_id, timestamp, target y covariables.
    historical_records = []
    for row in historical_rows:
        if row.price is None:
            raise HTTPException(
                status_code=400,
                detail="Missing price data for Chronos"
            )

        exog_values = {
            "demand_forecast": row.demand_forecast,
            "wind_forecast": row.wind_forecast,
            "solar_forecast": row.solar_forecast,
            "hydro_programmed": row.hydro_programmed,
        }

        # Chronos con covariables exige que el histórico tenga también esas variables completas.
        if any(value is None for value in exog_values.values()):
            raise HTTPException(
                status_code=400,
                detail="Missing exogenous data for Chronos"
            )
        # Cada fila del histórico es un punto de la serie temporal con su timestamp, valor objetivo (price) y covariables.
        historical_records.append(
            {
                "item_id": "price",
                "timestamp": row.timestamp,
                "target": row.price,
                **exog_values,
            }
        )
    # Convierte el histórico a DataFrame y ordena por timestamp. Chronos requiere que el histórico esté ordenado cronológicamente.
    historical_df = pd.DataFrame(historical_records)
    historical_df["timestamp"] = pd.to_datetime(historical_df["timestamp"])
    historical_df = historical_df.sort_values("timestamp")
    historical_df = historical_df.dropna(subset=["target"])

    if historical_df.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid historical data available for Chronos"
        )

    # Convierte el histórico al formato TimeSeriesDataFrame que necesita AutoGluon.
    ts_df = TimeSeriesDataFrame.from_data_frame(
        historical_df,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    # Recupera las 24 horas futuras, necesarias para pasar known_covariates a Chronos.
    future_end = requested_date + timedelta(hours=24)
    future_rows = get_historical_data_between(
        db=db,
        start=requested_date + timedelta(hours=1),
        end=future_end
    )

    # Quitar timestamps repetidos para quedarme con una sola fila por hora.
    future_by_ts: dict[datetime, any] = {}
    for row in future_rows:
        if row.timestamp not in future_by_ts:
            future_by_ts[row.timestamp] = row
    future_rows = list(future_by_ts.values())

    # Chronos necesita covariables futuras para las 24 horas del horizonte.
    if len(future_rows) != 24:
        raise HTTPException(
            status_code=400,
            detail="Not enough future covariates for Chronos (need 24 hours)"
        )

    # Prepara las covariables futuras conocidas, sin target porque el target futuro es lo que queremos predecir.
    future_records = []
    for row in future_rows:
        exog_values = {
            "demand_forecast": row.demand_forecast,
            "wind_forecast": row.wind_forecast,
            "solar_forecast": row.solar_forecast,
            "hydro_programmed": row.hydro_programmed,
        }

        if any(value is None for value in exog_values.values()):
            raise HTTPException(
                status_code=400,
                detail="Missing future exogenous data for Chronos"
            )

        future_records.append(
            {
                "item_id": "price",
                "timestamp": row.timestamp,
                **exog_values,
            }
        )
    # Convierte las covariables futuras a DataFrame y ordena por timestamp.
    known_covariates_df = pd.DataFrame(future_records)
    known_covariates_df["timestamp"] = pd.to_datetime(known_covariates_df["timestamp"])
    known_covariates_df = known_covariates_df.sort_values("timestamp")

    # Convierte las covariables futuras al formato AutoGluon.
    known_covariates_ts = TimeSeriesDataFrame.from_data_frame(
        known_covariates_df,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    # Carga el predictor Chronos/AutoGluon y genera la predicción.
    try:
        predictor = load_chronos_predictor()
        predictions = predictor.predict(
            data=ts_df,
            known_covariates=known_covariates_ts
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chronos prediction failed: {exc}"
        ) from exc

    # Comprueba que AutoGluon haya devuelto predicciones.
    if predictions is None or predictions.empty:
        raise HTTPException(
            status_code=500,
            detail="Chronos prediction returned empty output"
        )

    # Selecciona la serie con item_id='price'.
    try:
        item_predictions = predictions.loc["price"]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chronos output missing item_id 'price': {exc}"
        ) from exc

    predictions_df = item_predictions.reset_index()

    # Se usa la columna mean como predicción puntual de Chronos.
    if "mean" not in predictions_df.columns:
        raise HTTPException(
            status_code=500,
            detail="Chronos output does not include mean predictions"
        )

    predictions_df["timestamp"] = pd.to_datetime(predictions_df["timestamp"])
    predictions_df = predictions_df.sort_values("timestamp").head(24)

    if len(predictions_df) < 24:
        raise HTTPException(
            status_code=400,
            detail="Chronos did not return 24 future steps"
        )

    # Convierte la salida de AutoGluon al schema común de la API.
    forecast = [
        ForecastPointResponse(
            timestamp=row["timestamp"].to_pydatetime(),
            value=float(row["mean"])
        )
        for _, row in predictions_df.iterrows()
    ]

    return ForecastResponse(
        model="chronos",
        model_type="foundation_model",
        requested_date=requested_date,
        horizon_hours=24,
        forecast=forecast
    )
