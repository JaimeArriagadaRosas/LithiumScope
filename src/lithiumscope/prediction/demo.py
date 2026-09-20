from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import DATA_DIR, PROJECT_ROOT
from lithiumscope.model_1.steps.step_01_load_data import load_data


def demonstration_cases_path() -> Path:
    raw = str(load_config("prediction")["demonstration"]["cases_path"])
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_demonstration_cases() -> pd.DataFrame:
    path = demonstration_cases_path()
    frame = load_data(path, quiet=True)
    required = {"case_id", "Li_icpms", "Longitude", "Latitude"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            "El conjunto de demostración está incompleto: "
            + ", ".join(sorted(missing))
        )
    return frame


def demonstration_imagery_cache() -> Path:
    raw = str(load_config("prediction")["demonstration"]["imagery_cache_dir"])
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def audit_demo_overlap(cases: pd.DataFrame) -> dict:
    training_path = DATA_DIR / "raw" / "model_1" / "Mamani09_Table_DR2.csv"
    base = {
        "training_dataset_path": str(training_path),
        "sample_id_matches": None,
        "coordinate_matches": None,
    }
    if not training_path.is_file():
        return {
            **base,
            "status": (
                "fuente externa por procedencia; no fue posible verificar "
                "solapamiento exacto porque el dataset local de entrenamiento no está disponible"
            ),
        }

    training = load_data(training_path, quiet=True)
    sample_column = next(
        (name for name in ("Sample", "sample_id", "sample", "ID", "id") if name in training.columns),
        None,
    )
    demo_ids = set(cases["case_id"].astype(str).str.strip())
    training_ids = (
        set(training[sample_column].dropna().astype(str).str.strip())
        if sample_column is not None
        else set()
    )
    sample_matches = len(demo_ids & training_ids)

    lon_column = next(
        (name for name in ("Longitude (X)", "Longitude", "longitude", "lon") if name in training.columns),
        None,
    )
    lat_column = next(
        (name for name in ("Latitude (Y)", "Latitude", "latitude", "lat") if name in training.columns),
        None,
    )
    coordinate_matches = 0
    if lon_column is not None and lat_column is not None:
        training_coords = set(
            zip(
                pd.to_numeric(training[lon_column], errors="coerce").round(4),
                pd.to_numeric(training[lat_column], errors="coerce").round(4),
            )
        )
        demo_coords = set(
            zip(
                pd.to_numeric(cases["Longitude"], errors="coerce").round(4),
                pd.to_numeric(cases["Latitude"], errors="coerce").round(4),
            )
        )
        training_coords = {
            pair for pair in training_coords if pd.notna(pair[0]) and pd.notna(pair[1])
        }
        demo_coords = {
            pair for pair in demo_coords if pd.notna(pair[0]) and pd.notna(pair[1])
        }
        coordinate_matches = len(training_coords & demo_coords)

    if sample_matches == 0 and coordinate_matches == 0:
        status = "sin coincidencias exactas detectadas contra el dataset local de entrenamiento"
    else:
        status = "posible solapamiento detectado; las métricas no deben tratarse como validación independiente"

    return {
        **base,
        "status": status,
        "sample_id_matches": sample_matches,
        "coordinate_matches": coordinate_matches,
    }
