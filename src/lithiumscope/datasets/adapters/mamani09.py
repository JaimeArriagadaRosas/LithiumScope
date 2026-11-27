"""Adapt the registered Mamani09 public mirror without asserting official equivalence."""
from pathlib import Path
import pandas as pd
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger
from lithiumscope.datasets.adapters.contracts import AdaptedSource
from lithiumscope.datasets.readers.tabular import load_data
from lithiumscope.datasets.validation.spatial.coordinates import assess_coordinate_aliases, coordinate_summary
from lithiumscope.datasets.validation.chemistry.analytical import parse_analytical_value

def adapt_mamani09(path: Path, *, lithium_unit="unknown"):
    if lithium_unit not in {"unknown", "ppm", "mg/kg"}:
        raise ValueError("Unsupported Mamani09 lithium unit")
    path = Path(path)
    digest = file_sha256(path)
    original = load_data(path, quiet=True)
    if "Li_icpms" not in original or not original.columns.is_unique:
        raise ValueError("Mamani09 mirror requires Li_icpms and unique columns")
    reserved = {"source_row_position", "source_dataset", "source_file", "source_li_raw",
                "source_li_unit", "source_li_censor", "source_li_limit", "source_coordinate_status"}
    if reserved & set(original) or any(str(name).startswith("source_coordinate_original_") for name in original):
        raise ValueError("Source provenance columns already present")
    frame, coordinates = assess_coordinate_aliases(original)
    if not {"Longitude", "Latitude"} <= set(frame):
        raise ValueError("Mamani09 mirror lacks coordinate columns")
    parsed = [parse_analytical_value(value, decimal_comma=False) for value in original.Li_icpms]
    frame["source_li_raw"] = original.Li_icpms
    frame["source_li_censor"] = [kind for _, kind, _ in parsed]
    frame["source_li_limit"] = [limit for _, _, limit in parsed]
    # Censored assays are not measured point labels.
    frame["Li_icpms"] = pd.to_numeric([value if kind is None else None for value, kind, _ in parsed], errors="coerce")
    frame["source_li_unit"] = lithium_unit
    frame["source_coordinate_status"] = coordinates.coordinate_status
    for name in coordinates.columns:
        if name.startswith("original_"):
            frame["source_coordinate_" + name] = coordinates[name]
    frame["source_row_position"] = range(len(frame))
    frame["source_dataset"] = "Mamani09_public_mirror"
    frame["source_file"] = path.name
    if file_sha256(path) != digest:
        raise ValueError("Mamani09 source changed during adaptation")
    evidence = {"source_sha256": digest, "rows": len(frame), "official_equivalence_certified": False,
                "training_label_certified": False, "coordinate_quality": coordinate_summary(coordinates)}
    get_logger("datasets.adapters.mamani09").info("Adapted mirror rows=%d source=%s", len(frame), path.name)
    return AdaptedSource(frame, evidence)
