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

# Inicializa la aplicación FastAPI 
app = FastAPI(
    title=settings.APP_NAME, 
    version=settings.APP_VERSION,
    description="API para consulta histórica y predicción del mercado eléctrico español"
)
# Configuro CORS para que el frontend pueda hacer peticiones al backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Registro los routers, la API queda organizada por módulos: histórico, predicción, modelos y métricas.
app.include_router(health.router, prefix="/api/v1")
app.include_router(historical.router, prefix="/api/v1")
app.include_router(forecast.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(metrics.router, prefix="/api/v1")

logger = logging.getLogger("uvicorn")

# Función para cargar datos iniciales en la base de datos si está vacía. Esto se ejecuta al iniciar la aplicación.
def _seed_market_data_if_empty() -> None:
    Base.metadata.create_all(bind=engine)

    db = SessionLocal() # Creo una sesión de base de datos para verificar si ya hay datos en la tabla market_data
    try:
        has_row = db.query(MarketData.id).first()
    finally:
        db.close()

    if has_row: # Si ya hay datos, no hago nada
        return

    data_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "spot_es_with_exogenous.parquet"
    if not data_path.exists(): # Si el archivo de datos no existe, logueo una advertencia y salgo de la función
        logger.warning("No se encontro el archivo de datos: %s", data_path)
        return

    try: # Si el archivo existe, lo cargo con pandas, me quedo las columnas relevantes y convierto timestamp a formato datetime sin zona horaria.
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

        records = df.to_dict(orient="records") #
        db = SessionLocal()
        try:
            db.bulk_insert_mappings(MarketData, records) # Inserto los datos en la tabla market_data
            logger.info("Inserted %s rows into market_data", len(records)) 
        finally:
            db.close()
    except Exception:
        logger.exception("Error al cargar datos iniciales en market_data")


@app.on_event("startup") # Se ejecuta al iniciar la aplicación. Cargar datos iniciales en la base de datos si está vacía.
def on_startup() -> None:
    _seed_market_data_if_empty()


@app.get("/") # Verificar que la API está corriendo.
def root():
    return {"message": "Electricity Market Forecast API is running"}
