"""Persistent provenance identities, distinct from physical sample equivalence."""
from __future__ import annotations

import hashlib
import json

import pandas as pd


def observation_ids(frame: pd.DataFrame, *, source: str, source_sha256: str,
                    row_position: str = "source_row_position") -> pd.Series:
    if not source.strip() or len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256.lower()):
        raise ValueError("A source namespace and SHA-256 are required")
    if row_position not in frame or not frame.columns.is_unique:
        raise ValueError("Original row positions required")
    raw = pd.to_numeric(frame[row_position], errors="coerce")
    if raw.isna().any() or (raw < 0).any() or (raw % 1 != 0).any() or raw.duplicated().any():
        raise ValueError("Original row positions must be unique nonnegative integers")
    return raw.map(lambda position: hashlib.sha256(json.dumps(
        [source, source_sha256.lower(), int(position)], ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()).rename("observation_id")
