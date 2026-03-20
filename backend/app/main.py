from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import health
from app.core.config import settings

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


@app.get("/")
def root():
    return {"message": "Electricity Market Forecast API is running"}