from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from fastapi import HTTPException
from sqlalchemy.orm import Session
from autogluon.timeseries import TimeSeriesDataFrame

from app.application.services.historical_service import (
    get_historical_data_between,
    get_historical_data_range,
    get_previous_24_hours,
)
from app.application.services.model_service import is_supported_model
from app.schemas.forecast import ForecastPointResponse, ForecastResponse
from app.infrastructure.ml.xgboost_loader import (
    load_xgboost_model,
    load_xgboost_multistep_complete_model,
    load_xgboost_multistep_minimal_model,
)
from app.infrastructure.ml.chronos_loader import load_chronos_predictor


def generate_dummy_forecast(
    requested_date: datetime,
    model: str
) -> ForecastResponse:
    if not is_supported_model(model):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not supported"
        )

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


def generate_seasonal_naive_forecast(
    db: Session,
    requested_date: datetime
) -> ForecastResponse:

    if requested_date.tzinfo is not None:
        requested_date = requested_date.replace(tzinfo=None)

    data_range = get_historical_data_range(db)
    if data_range["start"] is None or data_range["end"] is None:
        raise HTTPException(
            status_code=400,
            detail="No historical data available"
        )

    if data_range["start"].tzinfo is not None:
        data_range["start"] = data_range["start"].replace(tzinfo=None)
    if data_range["end"].tzinfo is not None:
        data_range["end"] = data_range["end"].replace(tzinfo=None)

    if requested_date.minute != 0 or requested_date.second != 0:
        raise HTTPException(
            status_code=400,
            detail="Date must be aligned to full hour (e.g., 2022-01-01T00:00:00)"
        )

    if requested_date <= data_range["start"]:
        raise HTTPException(
            status_code=400,
            detail="Requested date is too early"
        )

    if requested_date > data_range["end"]:
        raise HTTPException(
            status_code=400,
            detail="Requested date is beyond available data"
        )

    previous_24h = get_previous_24_hours(db=db, requested_date=requested_date)

    if len(previous_24h) != 24:
        raise HTTPException(
            status_code=400,
            detail="Not enough historical data: need previous 24 hours"
        )

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
XGBOOST_INPUT_WINDOW = 168
XGBOOST_HORIZON = 24


def _build_price_window_168(
    requested_date: datetime,
    price_by_ts: dict[datetime, float],
) -> np.ndarray:
    timestamps = [
        requested_date - timedelta(hours=offset)
        for offset in range(XGBOOST_INPUT_WINDOW - 1, -1, -1)
    ]

    if any(ts not in price_by_ts for ts in timestamps):
        raise HTTPException(
            status_code=400,
            detail="Not enough historical price data for XGBoost multi-step (need 168 hours)"
        )

    price_window = np.asarray([price_by_ts[ts] for ts in timestamps], dtype=float)

    if price_window.shape[0] != XGBOOST_INPUT_WINDOW:
        raise HTTPException(
            status_code=400,
            detail="Invalid historical price window for XGBoost multi-step"
        )

    return price_window


def _future_timestamps(
    requested_date: datetime,
    horizon: int = XGBOOST_HORIZON,
) -> list[datetime]:
    return [requested_date + timedelta(hours=i) for i in range(1, horizon + 1)]


def _calendar_features_for_timestamps(
    timestamps: list[datetime],
) -> np.ndarray:
    rows = []
    for ts in timestamps:
        hour = ts.hour
        dayofweek = ts.weekday()
        month = ts.month
        is_weekend = 1 if dayofweek >= 5 else 0

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

    return np.asarray(rows, dtype=float)


def _future_exogenous_features(
    timestamps: list[datetime],
    exog_by_ts: dict[datetime, dict[str, float]],
) -> np.ndarray | None:
    values = []
    for ts in timestamps:
        exog = exog_by_ts.get(ts)
        if exog is None:
            return None

        current_values = [
            exog.get("demand_forecast"),
            exog.get("wind_forecast"),
            exog.get("solar_forecast"),
            exog.get("hydro_programmed"),
        ]
        if any(value is None for value in current_values):
            return None

        values.extend(float(value) for value in current_values)

    return np.asarray(values, dtype=float)


def _build_xgboost_multistep_input(
    requested_date: datetime,
    price_by_ts: dict[datetime, float],
    exog_by_ts: dict[datetime, dict[str, float]],
) -> tuple[np.ndarray, str]:
    price_window = _build_price_window_168(requested_date, price_by_ts)
    future_ts = _future_timestamps(requested_date)
    calendar_features = _calendar_features_for_timestamps(future_ts)
    exogenous_features = _future_exogenous_features(future_ts, exog_by_ts)

    if exogenous_features is not None:
        X = np.concatenate([price_window, calendar_features, exogenous_features])
        variant = "complete"
    else:
        X = np.concatenate([price_window, calendar_features])
        variant = "minimal"

    return X.reshape(1, -1), variant


def _build_xgboost_features_for_timestamp(
    ts: datetime,
    price_by_ts: dict[datetime, float],
    exog_by_ts: dict[datetime, dict[str, float]],
) -> pd.DataFrame:
    lag_1_ts = ts - timedelta(hours=1)
    lag_24_ts = ts - timedelta(hours=24)
    lag_168_ts = ts - timedelta(hours=168)

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

def generate_xgboost_forecast(
    db: Session,
    requested_date: datetime
) -> ForecastResponse:
    if requested_date.tzinfo is not None:
        requested_date = requested_date.replace(tzinfo=None)
    if requested_date.minute != 0 or requested_date.second != 0:
        raise HTTPException(
            status_code=400,
            detail="Date must be aligned to full hour (e.g., 2022-01-01T00:00:00)"
        )

    start = requested_date - timedelta(hours=XGBOOST_INPUT_WINDOW - 1)
    end = requested_date + timedelta(hours=XGBOOST_HORIZON)
    rows = get_historical_data_between(db, start, end)

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No historical data available for XGBoost"
        )

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
        if ts <= requested_date:
            price_by_ts[ts] = row.price

    if requested_date not in price_by_ts:
        raise HTTPException(
            status_code=400,
            detail="Requested date not present in historical data for XGBoost"
        )

    X, variant = _build_xgboost_multistep_input(requested_date, price_by_ts, exog_by_ts)

    if variant == "complete":
        model = load_xgboost_multistep_complete_model()
    else:
        model = load_xgboost_multistep_minimal_model()

    pred = model.predict(X)[0]
    if len(pred) != XGBOOST_HORIZON:
        raise HTTPException(
            status_code=500,
            detail=f"XGBoost multi-step model returned {len(pred)} steps, expected {XGBOOST_HORIZON}"
        )

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


def generate_chronos_forecast(
    db: Session,
    requested_date: datetime
) -> ForecastResponse:
    if requested_date.tzinfo is not None:
        requested_date = requested_date.replace(tzinfo=None)
    if requested_date.minute != 0 or requested_date.second != 0:
        raise HTTPException(
            status_code=400,
            detail="Date must be aligned to full hour (e.g., 2022-01-01T00:00:00)"
        )

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

    # datasets con duplicados por hora.
    hist_by_ts: dict[datetime, any] = {}
    for row in historical_rows:
        if row.timestamp not in hist_by_ts:
            hist_by_ts[row.timestamp] = row
    historical_rows = list(hist_by_ts.values())

    if historical_rows[-1].timestamp < requested_date:
        raise HTTPException(
            status_code=400,
            detail="Requested date not present in historical data for Chronos"
        )

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

        if any(value is None for value in exog_values.values()):
            raise HTTPException(
                status_code=400,
                detail="Missing exogenous data for Chronos"
            )

        historical_records.append(
            {
                "item_id": "price",
                "timestamp": row.timestamp,
                "target": row.price,
                **exog_values,
            }
        )

    historical_df = pd.DataFrame(historical_records)
    historical_df["timestamp"] = pd.to_datetime(historical_df["timestamp"])
    historical_df = historical_df.sort_values("timestamp")
    historical_df = historical_df.dropna(subset=["target"])

    if historical_df.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid historical data available for Chronos"
        )

    ts_df = TimeSeriesDataFrame.from_data_frame(
        historical_df,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    future_end = requested_date + timedelta(hours=24)
    future_rows = get_historical_data_between(
        db=db,
        start=requested_date + timedelta(hours=1),
        end=future_end
    )

    # Dedupe por timestamp (datasets con duplicados por hora).
    future_by_ts: dict[datetime, any] = {}
    for row in future_rows:
        if row.timestamp not in future_by_ts:
            future_by_ts[row.timestamp] = row
    future_rows = list(future_by_ts.values())

    if len(future_rows) != 24:
        raise HTTPException(
            status_code=400,
            detail="Not enough future covariates for Chronos (need 24 hours)"
        )

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

    known_covariates_df = pd.DataFrame(future_records)
    known_covariates_df["timestamp"] = pd.to_datetime(known_covariates_df["timestamp"])
    known_covariates_df = known_covariates_df.sort_values("timestamp")

    known_covariates_ts = TimeSeriesDataFrame.from_data_frame(
        known_covariates_df,
        id_column="item_id",
        timestamp_column="timestamp"
    )
    #lo anterior es preparación de datos para Chronos: cargar histórico, verificar que cubre el requested_date, preparar dataframe con target y covariables, cargar covariables futuras para las 24 horas siguientes.
    #aquí llamamos al modelo de Chronos para que haga la predicción, y luego procesamos la salida para devolverla en el formato esperado por la API.
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

    if predictions is None or predictions.empty:
        raise HTTPException(
            status_code=500,
            detail="Chronos prediction returned empty output"
        )

    try:
        item_predictions = predictions.loc["price"]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chronos output missing item_id 'price': {exc}"
        ) from exc

    predictions_df = item_predictions.reset_index()

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
