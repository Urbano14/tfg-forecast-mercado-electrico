import joblib

from app.core.config import settings


def load_xgboost_model():
    print("Loading model from:", settings.XGBOOST_MODEL_PATH)
    return joblib.load(settings.XGBOOST_MODEL_PATH)