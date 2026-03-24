from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    

    APP_NAME: str = "Electricity Market Forecast API"
    APP_VERSION: str = "0.1.0"

    DATABASE_URL: str
    FRONTEND_URL: str = "http://localhost:5173"

    DATA_PATH: str = "../data/processed/spot_es_with_exogenous.parquet"
    MODELS_PATH: str = "../models/"

    XGBOOST_MODEL_PATH: str = str( BASE_DIR / "models" / "xgboost" / "xgboost_optuna_exogenous.pkl")

    ESIOS_TOKEN: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
