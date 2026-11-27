"""Read-only audit of raw/filtered GEOROC exports, before numeric harmonization."""
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.readers.georoc_raw_reader import detect_georoc_encoding
from lithiumscope.datasets.adapters.georoc.material import parse_georoc_material
from lithiumscope.datasets.validation.chemistry.analytical import parse_analytical_value


def inspect_export(path: Path):
    encoding = detect_georoc_encoding(path)
    frame = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False, low_memory=False)
    required = {"UNIQUE_ID", "MATERIAL", "LI(PPM)", "CITATIONS"}
    if not required <= set(frame):
        raise ValueError(f"Missing provenance fields: {sorted(required - set(frame))}")
    chemical = [c for c in frame if c.endswith(("(WT%)", "(PPM)"))]
    raw = frame[chemical].apply(lambda v: v.str.strip())
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    material = frame["MATERIAL"].map(lambda v: parse_georoc_material(v).code or "unknown")
    unusual = (raw != "") & numeric.isna()
    records = []
    for column in chemical:
        for idx in raw.index[unusual[column]]:
            value, kind, limit = parse_analytical_value(raw.loc[idx, column], decimal_comma=False)
            records.append({"unique_id": frame.loc[idx, "UNIQUE_ID"], "column": column,
                            "original": raw.loc[idx, column], "kind": kind or "unparseable",
                            "limit": limit, "substitution": value if kind and pd.notna(value) else None})
    li = numeric["LI(PPM)"]
    info = {"path": str(path), "sha256": file_sha256(path), "encoding": encoding,
            "rows": len(frame), "columns": len(frame.columns), "material_counts": material.value_counts().to_dict(),
            "li_unit": "ppm", "li_method": "unknown", "coordinate_crs": "unknown",
            "li_informed": int(raw["LI(PPM)"].ne("").sum()), "li_finite": int(np.isfinite(li).sum()),
            "li_zero": int(li.eq(0).sum()), "li_negative": int(li.lt(0).sum()),
            "unique_id_missing": int(frame.UNIQUE_ID.str.strip().eq("").sum()),
            "unique_id_duplicates": int(frame.UNIQUE_ID.duplicated().sum()),
            "citations_missing": int(frame.CITATIONS.str.strip().eq("").sum()),
            "non_numeric_analytical_tokens": len(records),
            "iron_form_counts": {c: int(raw[c].ne("").sum()) for c in
                                 ("FEOT(WT%)", "FE2O3(WT%)", "FEO(WT%)", "FE2O3T(WT%)") if c in raw}}
    extremes = frame.loc[li.notna(), [c for c in ("UNIQUE_ID", "SAMPLE NAME", "CITATIONS", "ROCK CLASS",
                         "LATITUDE MIN", "LATITUDE MAX", "LONGITUDE MIN", "LONGITUDE MAX") if c in frame]].copy()
    extremes["Li_ppm"] = li.loc[extremes.index]
    extremes = pd.concat([extremes.loc[extremes.Li_ppm.eq(0)], extremes.nlargest(20, "Li_ppm")]).drop_duplicates()
    return info, frame, material, records, extremes


def compare_exports(original: Path, filtered: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    before, raw, material, raw_tokens, _ = inspect_export(original)
    after, selected, _, selected_tokens, extremes = inspect_export(filtered)
    expected = raw.loc[material.eq("WR")].copy()
    selected = selected.copy()
    # Historical pandas export serialized integral numeric IDs with .0.
    # Preserve original strings and only normalize the proven representation.
    for cohort in (expected, selected):
        cohort["_identity"] = cohort.UNIQUE_ID.str.strip().str.replace(r"^([1-9][0-9]*)\.0+$", r"\1", regex=True)
    ids = set(expected._identity)
    actual = set(selected._identity)
    comparable = sorted(ids & actual)
    columns = [c for c in raw if c in selected and c.endswith(("(PPM)", "(WT%)"))]
    differences = None
    if comparable and not (expected._identity.duplicated().any() or selected._identity.duplicated().any()):
        left = expected.set_index("_identity").loc[comparable, columns].apply(pd.to_numeric, errors="coerce")
        right = selected.set_index("_identity").loc[comparable, columns].apply(pd.to_numeric, errors="coerce")
        differences = int((~np.isclose(left.to_numpy(float), right.to_numpy(float), rtol=1e-12,
                                       atol=1e-12, equal_nan=True)).sum())
    return {"original": before, "filtered": after, "expected_whole_rock_rows": len(expected),
            "missing_whole_rock_ids": sorted(ids - actual), "unexpected_filtered_ids": sorted(actual - ids),
            "matched_ids": len(comparable), "numeric_chemistry_cell_differences": differences,
            "numeric_chemistry_columns_compared": len(columns),
            "limitation": "Identity by source ID; analytical technique, zero semantics, CRS and physical duplicate identity are not certified."}, pd.DataFrame(raw_tokens + selected_tokens), extremes
