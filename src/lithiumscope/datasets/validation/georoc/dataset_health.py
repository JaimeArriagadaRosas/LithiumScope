"""Provider-specific export health; presence is not scientific eligibility."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from lithiumscope.datasets.adapters.georoc.schema import infer_georoc_column_map
from lithiumscope.datasets.readers.georoc_raw_reader import detect_georoc_encoding
from lithiumscope.datasets.validation.chemistry.analytical import parse_analytical_value


@dataclass(frozen=True)
class DatasetHealth:
    rows: int
    columns: int
    li_nonempty: int
    valid_coordinate_pairs: int
    complete_li_coordinate_rows: int
    out_of_range_coordinates: int
    duplicate_rows: int
    li_numeric_valid: int = 0
    li_quantified_censored: int = 0
    li_invalid_or_unquantified: int = 0
    complete_numeric_li_coordinate_rows: int = 0

    def as_log_fields(self) -> dict[str, int]:
        return asdict(self)


def assess_dataset_health(frame: pd.DataFrame) -> DatasetHealth:
    if frame.empty:
        raise RuntimeError("La exportación GEOROC está vacía.")
    mapping = infer_georoc_column_map(frame.columns)
    if not frame.columns.is_unique:
        raise ValueError("Ambiguous GEOROC columns")
    by_target = {target: raw for raw, target in mapping.items()}
    selected = {}
    for target, options in (("LI", ("Li_icpms",)), ("LONGITUDE", ("Longitude", "Longitude_min")),
                            ("LATITUDE", ("Latitude", "Latitude_min"))):
        matches = [name for name in options if name in by_target]
        if matches:
            selected[target] = by_target[matches[0]]
    missing = [name for name in ("LI", "LONGITUDE", "LATITUDE") if name not in selected]
    if missing:
        raise RuntimeError("La exportación obtenida no corresponde al contrato LithiumScope. Faltan: " + ", ".join(missing))
    li = frame[selected["LI"]]
    present = li.notna() & li.astype(str).str.strip().ne("")
    numeric = pd.to_numeric(li, errors="coerce")
    valid = np.isfinite(numeric) & numeric.ge(0)
    def quantified(value):
        _, kind, limit = parse_analytical_value(value, decimal_comma=False)
        return kind in {"below", "above"} and limit is not None and np.isfinite(limit)
    censored = li.map(quantified)
    usable = valid | censored
    longitude = pd.to_numeric(frame[selected["LONGITUDE"]], errors="coerce")
    latitude = pd.to_numeric(frame[selected["LATITUDE"]], errors="coerce")
    numeric_coordinates = longitude.notna() & latitude.notna()
    coordinates = longitude.between(-180, 180) & latitude.between(-90, 90)
    complete = usable & coordinates
    if not usable.any():
        raise RuntimeError("La exportación GEOROC no contiene valores de litio utilizables.")
    if not coordinates.any():
        raise RuntimeError("La exportación GEOROC no contiene pares de coordenadas geográficas válidos.")
    if not complete.any():
        raise RuntimeError("La exportación GEOROC no contiene filas con litio y coordenadas válidas simultáneamente.")
    return DatasetHealth(rows=len(frame), columns=len(frame.columns), li_nonempty=int(present.sum()),
                         valid_coordinate_pairs=int(coordinates.sum()), complete_li_coordinate_rows=int(complete.sum()),
                         out_of_range_coordinates=int((numeric_coordinates & ~coordinates).sum()),
                         duplicate_rows=int(frame.duplicated().sum()), li_numeric_valid=int(valid.sum()),
                         li_quantified_censored=int(censored.sum()), li_invalid_or_unquantified=int((present & ~usable).sum()),
                         complete_numeric_li_coordinate_rows=int((valid & coordinates).sum()))


def assess_dataset_health_file(path) -> DatasetHealth:
    from pathlib import Path
    try:
        path = Path(path)
        frame = pd.read_csv(path, encoding=detect_georoc_encoding(path), low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"No se pudo evaluar la salud del dataset GEOROC descargado: {exc}") from exc
    return assess_dataset_health(frame)
