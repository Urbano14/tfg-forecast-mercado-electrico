from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import health
from app.api.routers import historical
from app.core.config import settings
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

@app.get("/")
def root():
    return {"message": "Electricity Market Forecast API is running"}
