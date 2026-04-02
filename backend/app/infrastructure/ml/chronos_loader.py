from autogluon.timeseries import TimeSeriesPredictor

from app.core.config import settings


def load_chronos_predictor():
    return TimeSeriesPredictor.load(settings.CHRONOS_MODEL_PATH)
