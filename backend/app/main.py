from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

from app.api.routers import health
from app.api.routers import historical
from app.core.config import settings
from app.core.database import SessionLocal, engine
from app.infrastructure.db.base import Base
from app.infrastructure.db.models import MarketData
from app.api.routers import forecast
from app.api.routers import models
from app.api.routers import metrics

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API para consulta histórica y predicción del mercado eléctrico español"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(historical.router, prefix="/api/v1")
app.include_router(forecast.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(metrics.router, prefix="/api/v1")

logger = logging.getLogger("uvicorn")


def _seed_market_data_if_empty() -> None:
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        has_row = db.query(MarketData.id).first()
    finally:
        db.close()

    if has_row:
        return

    data_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "spot_es_with_exogenous.parquet"
    if not data_path.exists():
        logger.warning("No se encontro el archivo de datos: %s", data_path)
        return

    try:
        import pandas as pd

        df = pd.read_parquet(data_path)
        df = df[
            [
                "timestamp",
                "price",
                "demand_forecast",
                "wind_forecast",
                "solar_forecast",
                "hydro_programmed",
            ]
        ].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

        records = df.to_dict(orient="records")
        db = SessionLocal()
        try:
            db.bulk_insert_mappings(MarketData, records)
            db.commit()
            logger.info("Inserted %s rows into market_data", len(records))
        finally:
            db.close()
    except Exception:
        logger.exception("Error al cargar datos iniciales en market_data")


@app.on_event("startup")
def on_startup() -> None:
    _seed_market_data_if_empty()


@app.get("/")
def root():
    return {"message": "Electricity Market Forecast API is running"}
