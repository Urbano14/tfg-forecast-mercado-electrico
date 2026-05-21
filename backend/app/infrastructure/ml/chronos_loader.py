import os
import pathlib
from functools import lru_cache
from pathlib import Path

from autogluon.timeseries import TimeSeriesPredictor

from app.core.config import settings


def _patch_windows_path_for_linux() -> None:
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def load_chronos_predictor() -> TimeSeriesPredictor:

    _patch_windows_path_for_linux()

    model_path = Path(settings.CHRONOS_MODEL_PATH).resolve()
    predictor_file = model_path / "predictor.pkl"

    if not predictor_file.exists():
        raise FileNotFoundError(
            f"Chronos predictor not found at {predictor_file}. "
            f"Configured CHRONOS_MODEL_PATH={settings.CHRONOS_MODEL_PATH}"
        )

    return TimeSeriesPredictor.load(str(model_path))
