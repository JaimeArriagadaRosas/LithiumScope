from __future__ import annotations

from dataclasses import dataclass
import re


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
