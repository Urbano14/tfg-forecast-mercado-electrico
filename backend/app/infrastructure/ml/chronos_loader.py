import os
import pathlib

from autogluon.timeseries import TimeSeriesPredictor

from app.core.config import settings


def load_chronos_predictor():
    # Permite cargar modelos serializados en Windows dentro de Linux.
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[attr-defined]
    return TimeSeriesPredictor.load(settings.CHRONOS_MODEL_PATH)
