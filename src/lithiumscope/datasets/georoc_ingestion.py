from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from lithiumscope.core.logger import get_logger

logger = get_logger("datasets.georoc_ingestion")


GEOROC_ALIASES: dict[str, tuple[str, ...]] = {
    "source_sample": (
        "SAMPLE NAME",
        "SAMPLE",
        "SAMPLE_ID",
        "SAMPLE ID",
    ),
    "Sample_type": (
        "TYPE OF MATERIAL",
        "SAMPLE TYPE",
    ),
    "Rock_type": (
        "ROCK NAME",
        "ROCK TYPE",
        "ROCK_NAME",
    ),
    "Longitude": (
        "LONGITUDE",
        "LONGITUDE (X)",
        "LON",
    ),
    "Latitude": (
        "LATITUDE",
        "LATITUDE (Y)",
        "LAT",
    ),
    "Age (Ma)": (
        "AGE(MA)",
        "AGE (MA)",
        "AGE",
    ),
    "SiO2": ("SIO2(WT%)", "SIO2 (WT%)", "SIO2"),
    "TiO2": ("TIO2(WT%)", "TIO2 (WT%)", "TIO2"),
    "Al2O3": ("AL2O3(WT%)", "AL2O3 (WT%)", "AL2O3"),
    "Fe2O3": (
        "FE2O3T(WT%)",
        "FE2O3T (WT%)",
        "FE2O3(WT%)",
        "FE2O3 (WT%)",
    ),
    "MnO": ("MNO(WT%)", "MNO (WT%)", "MNO"),
    "MgO": ("MGO(WT%)", "MGO (WT%)", "MGO"),
    "CaO": ("CAO(WT%)", "CAO (WT%)", "CAO"),
    "Na2O": ("NA2O(WT%)", "NA2O (WT%)", "NA2O"),
    "K2O": ("K2O(WT%)", "K2O (WT%)", "K2O"),
    "P2O5": ("P2O5(WT%)", "P2O5 (WT%)", "P2O5"),
    "Li_icpms": ("LI(PPM)", "LI (PPM)", "LI_PPM"),
    "Th_icpms": ("TH(PPM)", "TH (PPM)"),
    "U_icpms": ("U(PPM)", "U (PPM)"),
    "Rb_icpms": ("RB(PPM)", "RB (PPM)"),
    "Cs_icpms": ("CS(PPM)", "CS (PPM)"),
    "Nb_icpms": ("NB(PPM)", "NB (PPM)"),
    "Ta_icpms": ("TA(PPM)", "TA (PPM)"),
    "Pb_icpms": ("PB(PPM)", "PB (PPM)"),
    "Ba_icpms": ("BA(PPM)", "BA (PPM)"),
    "Sr_icpms": ("SR(PPM)", "SR (PPM)"),
    "Zr_icpms": ("ZR(PPM)", "ZR (PPM)"),
    "V_icpms": ("V(PPM)", "V (PPM)"),
    "Hf_icpms": ("HF(PPM)", "HF (PPM)"),
}


MAJOR_OXIDES = (
    "SiO2",
    "TiO2",
    "Al2O3",
    "Fe2O3",
    "MnO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "P2O5",
)


@dataclass(frozen=True)
class GeorocIngestionResult:
    path: Path
    files: tuple[str, ...]
    rows_read: int
    rows_kept: int
    rows_rejected_missing_core: int
    mapped_columns: dict[str, str]
    audit_path: Path


def _normalize(value: str) -> str:
    return "".join(
        character
        for character in value.strip().upper()
        if character not in {" ", "_", "-"}
    )


def infer_georoc_column_map(
    columns: Iterable[str],
) -> dict[str, str]:
    actual = {
        _normalize(str(column)): str(column)
        for column in columns
    }
    result: dict[str, str] = {}
    for target, aliases in GEOROC_ALIASES.items():
        for alias in aliases:
            match = actual.get(_normalize(alias))
            if match is not None:
                result[match] = target
                break
    return result


def _read_header(path: Path) -> list[str]:
    return [
        str(column)
        for column in pd.read_csv(
            path,
            nrows=0,
            encoding_errors="replace",
        ).columns
    ]


def _quality_sum(frame: pd.DataFrame) -> pd.Series:
    if not all(
        column in frame.columns
        for column in MAJOR_OXIDES
    ):
        return pd.Series(
            pd.NA,
            index=frame.index,
            dtype="Float64",
        )
    numeric = frame[list(MAJOR_OXIDES)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    complete = numeric.notna().all(axis=1)
    result = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="Float64",
    )
    result.loc[complete] = numeric.loc[
        complete
    ].sum(axis=1)
    return result


def _harmonize_chunk(
    chunk: pd.DataFrame,
    *,
    column_map: dict[str, str],
    source_name: str,
    source_file: str,
    allowed_material_types: tuple[str, ...],
    minimum_predictors: int,
) -> tuple[pd.DataFrame, int]:
    frame = chunk.rename(
        columns=column_map
    ).copy()

    if (
        allowed_material_types
        and "Sample_type" in frame.columns
    ):
        accepted = {
            value.strip().upper()
            for value in allowed_material_types
        }
        frame = frame[
            frame["Sample_type"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(accepted)
        ].copy()

    numeric_columns = [
        column
        for column in (
            "Longitude",
            "Latitude",
            "Age (Ma)",
            *MAJOR_OXIDES,
            "Li_icpms",
            "Th_icpms",
            "U_icpms",
            "Rb_icpms",
            "Cs_icpms",
            "Nb_icpms",
            "Ta_icpms",
            "Pb_icpms",
            "Ba_icpms",
            "Sr_icpms",
            "Zr_icpms",
            "V_icpms",
            "Hf_icpms",
        )
        if column in frame.columns
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    required = [
        column
        for column in (
            "Li_icpms",
            "Longitude",
            "Latitude",
        )
        if column in frame.columns
    ]
    before = len(frame)
    if len(required) != 3:
        return frame.iloc[0:0].copy(), before

    frame = frame.dropna(
        subset=[
            "Li_icpms",
            "Longitude",
            "Latitude",
        ]
    )
    frame = frame[
        frame["Longitude"].between(-180, 180)
        & frame["Latitude"].between(-90, 90)
    ].copy()

    predictor_columns = [
        column
        for column in (
            *MAJOR_OXIDES,
            "Th_icpms",
            "U_icpms",
            "Rb_icpms",
            "Cs_icpms",
            "Nb_icpms",
            "Ta_icpms",
            "Pb_icpms",
            "Ba_icpms",
            "Sr_icpms",
            "Zr_icpms",
            "V_icpms",
            "Hf_icpms",
        )
        if column in frame.columns
    ]
    if predictor_columns and minimum_predictors > 0:
        usable = frame[predictor_columns].notna().sum(axis=1)
        frame = frame[
            usable >= minimum_predictors
        ].copy()

    if "SUM (no water)" not in frame.columns:
        frame["SUM (no water)"] = _quality_sum(frame)

    frame["source_dataset"] = source_name
    frame["source_file"] = source_file
    if "source_sample" not in frame.columns:
        frame["source_sample"] = [
            f"{Path(source_file).stem}_{index}"
            for index in frame.index
        ]

    rejected = before - len(frame)
    return frame, rejected


def ingest_georoc_files(
    files: Iterable[Path],
    destination: Path,
    *,
    audit_path: Path | None = None,
    chunksize: int = 100_000,
    source_name: str = "GEOROC",
    allowed_material_types: tuple[str, ...] = (
        "WHOLE ROCK",
    ),
    minimum_predictors: int = 8,
) -> GeorocIngestionResult:
    paths = tuple(
        Path(path)
        for path in files
        if Path(path).is_file()
    )
    if not paths:
        raise ValueError(
            "No GEOROC CSV files were supplied."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    audit_destination = (
        audit_path
        or destination.with_suffix(".audit.json")
    )

    rows_read = 0
    rows_kept = 0
    rejected = 0
    mapped_union: dict[str, str] = {}
    wrote_header = False
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )
    temporary.unlink(missing_ok=True)

    try:
        for path in paths:
            header = _read_header(path)
            mapping = infer_georoc_column_map(header)
            mapped_union.update(mapping)

            required_targets = {
                "Li_icpms",
                "Longitude",
                "Latitude",
            }
            produced = set(mapping.values())
            missing = sorted(
                required_targets - produced
            )
            if missing:
                logger.warning(
                    "Skipping GEOROC file without core columns %s: %s",
                    missing,
                    path,
                )
                continue

            usecols = list(mapping.keys())
            for chunk in pd.read_csv(
                path,
                usecols=usecols,
                chunksize=max(1, int(chunksize)),
                low_memory=False,
                encoding_errors="replace",
            ):
                rows_read += len(chunk)
                harmonized, chunk_rejected = (
                    _harmonize_chunk(
                        chunk,
                        column_map=mapping,
                        source_name=source_name,
                        source_file=path.name,
                        allowed_material_types=(
                            allowed_material_types
                        ),
                        minimum_predictors=(
                            minimum_predictors
                        ),
                    )
                )
                rejected += chunk_rejected
                if harmonized.empty:
                    continue
                harmonized.to_csv(
                    temporary,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                )
                wrote_header = True
                rows_kept += len(harmonized)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    if not wrote_header:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            "GEOROC files produced no compatible rows."
        )

    temporary.replace(destination)

    audit = {
        "source": source_name,
        "files": [str(path) for path in paths],
        "rows_read": rows_read,
        "rows_kept": rows_kept,
        "rows_rejected_missing_core_or_density": rejected,
        "mapped_columns": mapped_union,
        "chunksize": chunksize,
        "allowed_material_types": list(
            allowed_material_types
        ),
        "minimum_predictors": minimum_predictors,
        "scientific_note": (
            "FE2O3T(WT%) is accepted as the total-iron "
            "counterpart for the current Fe2O3 predictor. "
            "The oxide-sum QC value is calculated only for rows "
            "with all ten major oxides available; incomplete sums "
            "are not fabricated. These assumptions remain traceable "
            "in the source mapping."
        ),
    }
    audit_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    audit_destination.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return GeorocIngestionResult(
        path=destination,
        files=tuple(str(path) for path in paths),
        rows_read=rows_read,
        rows_kept=rows_kept,
        rows_rejected_missing_core=rejected,
        mapped_columns=mapped_union,
        audit_path=audit_destination,
    )
