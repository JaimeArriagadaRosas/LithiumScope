"""Lossless accounting of auxiliary columns and bibliography in research CSVs."""
import csv
import re

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.exceptions import InputValidationError
from lithiumscope.datasets.readers.georoc_raw_reader import detect_georoc_encoding
from lithiumscope.datasets.validation.tabular.schema import validate_headers
from .schema import infer_georoc_column_map, MAJOR_OXIDES

_MARKER = re.compile(r"^(references|bibliography|citations|abbreviations)\s*:\s*$", re.I)


_NUMBER = re.compile(r"^[<>≤≥]?\s*[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?$")


def _measurement_positions(header, keep):
    mapping = infer_georoc_column_map([header[i] for i in keep])
    return [i for i in keep if (mapping.get(header[i], "") in MAJOR_OXIDES
        or mapping.get(header[i], "").endswith("_icpms")
        or mapping.get(header[i], "").startswith(("Longitude", "Latitude"))
        or header[i].strip().upper().replace(" ", "") in {"FEOT(WT%)", "FEOT"})]


def _has_measurement(row, positions):
    return any(i < len(row) and _NUMBER.fullmatch(row[i].strip()) for i in positions)


def _structure_error(source, line, header, row, reason):
    occupied = [i + 1 for i, value in enumerate(row) if value.strip()]
    preview = [(i + 1, value.strip()[:100]) for i, value in enumerate(row) if value.strip()][:6]
    return InputValidationError(
        f"{source.name}: {reason} at line {line}; expected={len(header)}, "
        f"actual={len(row)}, nonempty_columns={occupied[:20]}, preview={preview!r}")


def normalize_research_export(source, destination):
    """Write a temporary scientific table; reject unnamed populated data columns."""
    if source.resolve() == destination.resolve():
        raise ValueError("Research originals are immutable")
    digest = file_sha256(source)
    encoding = detect_georoc_encoding(source)
    footer, blank_lines, count = [], 0, 0
    section = None
    with source.open(encoding=encoding, newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        blank = {i for i, value in enumerate(header) if not value.strip()}
        validate_headers(header, allowed_blank=blank)
        keep = [i for i in range(len(header)) if i not in blank]
        validate_headers([header[i] for i in keep])
        measurement_positions = _measurement_positions(header, keep)
        nonempty = [0] * len(header)
        censored = [0] * len(header)
        with destination.open("x", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow([header[i] for i in keep])
            for row in reader:
                cells = [value.strip() for value in row if value.strip()]
                if not cells:
                    blank_lines += 1
                    continue
                # The historical GEOROC bibliography reader detects an explicit
                # marker in any nonempty cell, including a multi-cell footer row.
                marker = next((value for value in cells if _MARKER.fullmatch(value)), None)
                # Published 172-column partitions end with an inline legend:
                # "Abbreviations: MET: METAMORPHIC ROCK; PEG: PEGMATITE; ...".
                # Require the first cell, a short standalone row and an explicit
                # abbreviation pair. Never treat an arbitrary scientific note as a footer.
                if (marker is None and len(cells) == 1 and row[0].strip() == cells[0]
                        and re.match(r"^Abbreviations\s*:\s*[A-Z]{2,8}\s*:\s*\S", cells[0], re.I)):
                    marker = "Abbreviations:"
                has_measurement = _has_measurement(row, measurement_positions)
                if marker is not None:
                    if has_measurement:
                        raise _structure_error(source, reader.line_num, header, row,
                                               "Ambiguous footer marker with scientific values")
                    section = marker
                if section is not None:
                    if len(row) == len(header) and has_measurement:
                        raise _structure_error(source, reader.line_num, header, row,
                                               "Scientific-looking row after bibliography marker")
                    footer.append({"line": reader.line_num, "section": section, "cells": row})
                    continue
                if len(row) != len(header):
                    raise _structure_error(source, reader.line_num, header, row, "Malformed GEOROC row")
                if any(row[i].strip() for i in blank):
                    raise _structure_error(source, reader.line_num, header, row, "Unnamed GEOROC data column populated")
                for i in keep:
                    value = row[i].strip()
                    nonempty[i] += bool(value)
                    censored[i] += value.startswith(("<", ">"))
                writer.writerow([row[i] for i in keep])
                count += 1
    if not count or file_sha256(source) != digest:
        raise InputValidationError("Empty or changed GEOROC original")
    return {"original_sha256": digest, "encoding": encoding,
            "empty_auxiliary_columns_1based": [i+1 for i in sorted(blank)],
            "blank_records": blank_lines, "data_records": count, "footer": footer,
            "columns": [{"position_1based": i+1, "header": header[i],
                         "nonempty": nonempty[i], "empty": count-nonempty[i],
                         "limit_markers": censored[i]} for i in range(len(header))]}
