"""On-disk scoring artifacts. Written by the study, read by the local app.

The model inside is fit on the development listings only. The locked test
stays out of the file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

REPO = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO / "models" / "artifacts"


def artifact_path(city_key: str) -> Path:
    return ARTIFACT_ROOT / city_key / "model.joblib"


def save_artifact(city_key: str, payload: dict) -> Path:
    path = artifact_path(city_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path, compress=3)
    load_artifact.cache_clear()
    return path


@lru_cache(maxsize=8)
def load_artifact(city_key: str) -> dict:
    path = artifact_path(city_key)
    if not path.exists():
        raise FileNotFoundError(f"No model for {city_key}. Run: python -m evaluation.run_study")
    return joblib.load(path)


def available(city_key: str) -> bool:
    return artifact_path(city_key).is_file()
