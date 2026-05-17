from pathlib import Path

import joblib

from app.core.config import settings


BACKEND_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = BACKEND_DIR.parent
XGBOOST_MULTISTEP_MINIMAL_PATH = PROJECT_ROOT / "models/xgboost/xgboost_multistep_minimal.pkl"
XGBOOST_MULTISTEP_COMPLETE_PATH = PROJECT_ROOT / "models/xgboost/xgboost_multistep_complete.pkl"


def _load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"XGBoost model file not found: {path}")
    return joblib.load(path)


def load_xgboost_model():
    print("Loading model from:", settings.XGBOOST_MODEL_PATH)
    return joblib.load(settings.XGBOOST_MODEL_PATH)


def load_xgboost_multistep_minimal_model():
    return _load_model(XGBOOST_MULTISTEP_MINIMAL_PATH)


def load_xgboost_multistep_complete_model():
    return _load_model(XGBOOST_MULTISTEP_COMPLETE_PATH)
