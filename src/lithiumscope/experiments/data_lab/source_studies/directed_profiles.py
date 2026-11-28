"""Independent descriptive profiles for the registered M1 scientific supplements."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.core.scientific.eda_statistics import feature_profiles
from lithiumscope.results.eda_artifacts import complete_run, fail_run, find_compatible, new_run
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json
from . import chen, meixner, whole_rock, directed_intake

KEYS = (chen.SOURCE, directed_intake.MEIXNER, whole_rock.PUNA, whole_rock.MAC)
EXPECTED = {
    chen.SOURCE: {chen.NAME: chen.SHA},
    directed_intake.MEIXNER: meixner.ORIGINALS,
    whole_rock.PUNA: dict((whole_rock.FILES[whole_rock.PUNA], whole_rock.META)),
    whole_rock.MAC: dict((whole_rock.FILES[whole_rock.MAC],)),
}
_BANDS = [-np.inf, 0, 5, 10, 20, 40, 60, 80, 100, 126.12, 696, np.inf]


def _supplements(paths, tables):
    summaries, mentions = [], []
    for name, path in paths.items():
        directed_intake.check_file(path)
        sheets, cells, state = directed_intake.inspect(path)
        summaries.extend({"file": name, "inspection": state, **row} for row in sheets)
        mentions.extend({"file": name, **row} for row in cells)
    write_table(tables / "sheet_inventory.csv", pd.DataFrame(summaries))
    write_table(tables / "li_cell_inventory.csv", pd.DataFrame(mentions))


def _read_source(key, paths, tables, base):
    if key == chen.SOURCE:
        path = paths.get(chen.NAME)
        if path is None:
            raise FileNotFoundError(chen.NAME)
        with pd.ExcelFile(path, engine="openpyxl") as book:
            inventory = []
            for sheet in book.sheet_names:
                table = book.parse(sheet, header=None, dtype=object)
                inventory.append({"sheet": sheet, "rows": len(table), "columns": table.shape[1]})
                if sheet == chen.SHEET:
                    rows, columns, bands, summary = chen.profile(table)
            if chen.SHEET not in book.sheet_names:
                raise ValueError("Expected Chen measurement sheet unavailable")
        write_table(tables / "sheet_inventory.csv", pd.DataFrame(inventory))
        write_table(tables / "measurement_inventory.csv", columns)
        write_table(tables / "source_band_counts.csv", bands)
        return rows, summary
    _supplements(paths, tables)
    if key == directed_intake.MEIXNER:
        name = "126_2019_915_MOESM2_ESM.xls"
        if name not in paths:
            raise FileNotFoundError(name)
        rows = meixner.parse_a1(pd.read_excel(paths[name], sheet_name="Sheet1", header=None, engine="xlrd"))
        write_table(tables / "analysis_inventory.csv", rows)
        rows = rows.loc[~rows.repeat].rename(columns={"Li_ppm": "Li_icpms"}).copy()
        if base is not None:
            matches = meixner.match_names(rows, base, "historical_raw", "canonical")
            write_legacy_evidence_json(tables / "historical_identity_flags.json", matches)
        return rows, {"repeats_excluded_from_primary_profile": 7,
                      "canonical_predictors_available": ["SiO2"],
                      "matrix_note": "Lithology sections; standards and leaching are not rock Li labels"}
    name = whole_rock.FILES[key][0]
    if name not in paths:
        raise FileNotFoundError(name)
    puna = key == whole_rock.PUNA
    with pd.ExcelFile(paths[name], engine="openpyxl") as book:
        samples = book.parse("1 Samples" if puna else "2 Samples", header=None)
        data = book.parse("2 Data" if puna else "3 Data", header=None)
        rows, inventory, censored = whole_rock.parse_tables(samples, data, key)
    write_table(tables / "measurement_inventory.csv", inventory)
    write_legacy_evidence_json(tables / "censoring.json", censored)
    if base is not None:
        edges = whole_rock.duplicate_edges(rows, base, "historical_raw")
        write_legacy_evidence_json(tables / "historical_identity_flags.json", edges)
    return rows, {"material_review_required": True,
                  "publication_overlap_known": key == whole_rock.MAC}


def profile(key, paths, base_path, *, output_root):
    paths = {name: Path(path) for name, path in paths.items()}
    digests = {name: file_sha256(path) for name, path in paths.items()}
    base_digest = file_sha256(base_path) if base_path is not None else None
    code = {Path(m.__file__).name: file_sha256(Path(m.__file__))
            for m in (chen, meixner, whole_rock, directed_intake)}
    code["directed_profiles.py"] = file_sha256(Path(__file__))
    libraries = ("numpy", "pandas", "xlrd") if key == directed_intake.MEIXNER else ("numpy", "pandas", "openpyxl")
    signature = canonical_json_hash({"key": key, "files": digests, "base": base_digest,
        "code": code, "versions": {p: version(p) for p in libraries}})
    cached = find_compatible(output_root, signature)
    if cached is not None:
        return cached, True
    directory = new_run(output_root)
    try:
        tables = directory / "tables"
        base = pd.read_csv(base_path, low_memory=False) if base_path is not None else None
        rows, summary = _read_source(key, paths, tables, base)
        rows["source_dataset"] = key
        y = pd.to_numeric(rows["Li_icpms"], errors="coerce")
        valid = np.isfinite(y) & y.ge(0)
        # Chen profile deliberately retains only positive, observed Li; denominator
        # and discarded/reference-like rows are in its original source summary.
        censor = rows.get("source_li_censoring", pd.Series("", index=rows.index)).fillna("").astype(str)
        censored = ~censor.str.strip().str.lower().isin(("", "none", "exact", "not_censored", "nan"))
        summary.update(profile_scope="positive_observed_labels_only" if key == chen.SOURCE else "primary_review_rows",
                       censored_target_rows=int(censored.sum()),
                       uncensored_numeric_positive_li=int((valid & y.gt(0) & ~censored).sum()),
                       profile_rows=len(rows), finite_nonnegative_li=int(valid.sum()),
                       zero_li=int(y.eq(0).sum()), invalid_or_missing_li=int((~valid).sum()),
                       tail_above_40=int((valid & y.gt(40)).sum()),
                       tail_above_696=int((valid & y.gt(696)).sum()),
                       approved_rows=0)
        write_table(tables / "review_rows.csv", rows)
        write_table(tables / "feature_profiles.csv", feature_profiles(rows, list(rows.columns), stage="candidate"))
        counts = pd.cut(y[valid], _BANDS).value_counts(sort=False).rename_axis("band").reset_index(name="rows")
        write_table(tables / "lithium_bands.csv", counts)
        if any(file_sha256(path) != digests[name] for name, path in paths.items()) or (
                base_path is not None and file_sha256(base_path) != base_digest):
            raise ValueError("Original changed during directed EDA")
        return complete_run(directory, signature_value=signature, details={
            "model_group": "model_1", "scope": "directed_source_descriptive_review",
            "source_key": key, "source_sha256": digests, "base_sha256": base_digest,
            "summary": summary, "all_originals_available": set(paths) == set(EXPECTED[key]),
            "eligibility": "requires_review", "training_reusable": False}), False
    except Exception as exc:
        fail_run(directory, exc)
        raise
