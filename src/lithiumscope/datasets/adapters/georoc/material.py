from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
from .normalization import norm


_MATERIAL_RE = re.compile(
    r"^\s*(?:/\s*)?(?P<code>WR|GL)"
    r"(?:\s*\[(?P<batch>\d+)\])?\s*$",
    re.IGNORECASE,
)

_CANONICAL = {
    "WHOLE ROCK": "WR",
    "VOLCANIC GLASS": "GL",
}

_CANONICAL_LABELS = {
    "WR": "WHOLE ROCK",
    "GL": "VOLCANIC GLASS",
}


@dataclass(frozen=True)
class ParsedMaterial:
    raw: str
    code: str | None
    batch_id: str | None

    @property
    def known(self) -> bool:
        return self.code in {"WR", "GL"}


def parse_georoc_material(
    value: object,
) -> ParsedMaterial:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return ParsedMaterial(
            raw=raw,
            code=None,
            batch_id=None,
        )

    canonical = _CANONICAL.get(raw.upper())
    if canonical is not None:
        return ParsedMaterial(
            raw=raw,
            code=canonical,
            batch_id=None,
        )

    match = _MATERIAL_RE.fullmatch(raw)
    if match is None:
        return ParsedMaterial(
            raw=raw,
            code=None,
            batch_id=None,
        )

    return ParsedMaterial(
        raw=raw,
        code=match.group("code").upper(),
        batch_id=match.group("batch"),
    )



def canonical_material_label(
    code: str | None,
) -> str | None:
    if code is None:
        return None
    return _CANONICAL_LABELS.get(
        str(code).strip().upper()
    )


def material_code_for(
    value: object,
) -> str | None:
    return parse_georoc_material(
        value
    ).code


_MATERIAL_COLUMNS = (
    "TYPE OF MATERIAL",
    "SAMPLE TYPE",
    "MATERIAL",
)

@dataclass(frozen=True)
class MaterialFilterResult:
    frame: pd.DataFrame
    material_column: str
    rows_before: int
    rows_after: int
    materials_seen: tuple[str, ...]
    whole_rock_rows: int
    volcanic_glass_rows: int
    unknown_rows: int
    missing_rows: int

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


def _find_material_column(
    frame: pd.DataFrame,
) -> str | None:
    by_normalized = {}
    for column in frame.columns:
        normalized = norm(column)
        if normalized in by_normalized and normalized in {norm(c) for c in _MATERIAL_COLUMNS}:
            raise ValueError("Ambiguous GEOROC material header")
        by_normalized[normalized] = str(column)
    for candidate in _MATERIAL_COLUMNS:
        match = by_normalized.get(
            norm(candidate)
        )
        if match is not None:
            return match
    return None


def filter_whole_rock(
    frame: pd.DataFrame,
) -> MaterialFilterResult:
    material_column = _find_material_column(
        frame
    )
    if material_column is None:
        raise RuntimeError(
            "GEOROC no expuso una columna de material; "
            "no es posible demostrar el filtro WHOLE ROCK."
        )

    # If multiple material aliases are populated, they must agree on material identity.
    chosen = frame[material_column].astype("string").fillna("").str.strip()
    for alias in frame.columns:
        if alias == material_column or norm(alias) not in {norm(c) for c in _MATERIAL_COLUMNS}:
            continue
        alternate = frame[alias].astype("string").fillna("").str.strip()
        a = chosen.map(material_code_for)
        b = alternate.map(material_code_for)
        conflict = chosen.ne("") & alternate.ne("") & ~((a.notna() & a.eq(b)) | chosen.str.casefold().eq(alternate.str.casefold()))
        if conflict.any():
            raise ValueError("Conflicting GEOROC material aliases")
    material = (
        frame[material_column]
        .astype("string")
        .fillna("")
        .str.strip()
    )
    parsed = material.map(
        parse_georoc_material
    )
    codes = parsed.map(
        lambda item: item.code
    )

    whole_rock_mask = codes.eq("WR")
    volcanic_glass_mask = codes.eq("GL")
    missing_mask = material.eq("")
    unknown_mask = (
        ~missing_mask
        & codes.isna()
    )

    filtered = frame.loc[
        whole_rock_mask
    ].copy()

    materials_seen = tuple(
        sorted(
            {
                value
                for value in material.tolist()
                if value
            }
        )
    )
    if filtered.empty:
        preview = ", ".join(
            materials_seen[:12]
        )
        raise RuntimeError(
            "El filtro WHOLE ROCK dejó cero filas. "
            "Materiales observados: "
            + (preview or "(ninguno)")
        )

    return MaterialFilterResult(
        frame=filtered,
        material_column=material_column,
        rows_before=len(frame),
        rows_after=len(filtered),
        materials_seen=materials_seen,
        whole_rock_rows=int(
            whole_rock_mask.sum()
        ),
        volcanic_glass_rows=int(
            volcanic_glass_mask.sum()
        ),
        unknown_rows=int(
            unknown_mask.sum()
        ),
        missing_rows=int(
            missing_mask.sum()
        ),
    )


def select_material(frame, allowed_material_types):
    if "Sample_type" not in frame.columns and allowed_material_types:
        return frame.iloc[:0].copy()
    if "Sample_type" in frame.columns:
        frame["Sample_type_raw"] = (
            frame["Sample_type"]
            .astype("string")
            .fillna("")
            .str.strip()
        )
        parsed_material = frame[
            "Sample_type_raw"
        ].map(
            parse_georoc_material
        )
        frame["Sample_material_code"] = (
            parsed_material.map(
                lambda item: item.code
            )
        )
        frame["Sample_material_batch_id"] = (
            parsed_material.map(
                lambda item: item.batch_id
            )
        )
        frame["Sample_type"] = (
            frame["Sample_material_code"].map(
                canonical_material_label
            )
        )

        if allowed_material_types:
            accepted_codes = {
                parse_georoc_material(
                    value
                ).code
                for value in allowed_material_types
            }
            if None in accepted_codes:
                raise ValueError(
                    "allowed_material_types contiene "
                    "un material GEOROC no reconocido."
                )
            frame = frame[
                frame["Sample_material_code"].isin(
                    accepted_codes
                )
            ].copy()

    return frame
