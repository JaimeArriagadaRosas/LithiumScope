"""Lossless accounting of auxiliary columns and bibliography in research CSVs."""
import csv
import re

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.exceptions import InputValidationError
from lithiumscope.datasets.readers.georoc_raw_reader import detect_georoc_encoding
from lithiumscope.datasets.validation.tabular.schema import validate_headers

_MARKER = re.compile(r"^(references|bibliography|citations|abbreviations)\s*:\s*$", re.I)


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
                # Only a standalone, explicit section marker starts a footer.
                if len(cells) == 1 and _MARKER.fullmatch(cells[0]):
                    section = cells[0]
                if section is not None:
                    footer.append({"line": reader.line_num, "section": section, "cells": row})
                    continue
                if len(row) != len(header):
                    raise InputValidationError(f"Malformed GEOROC row at line {reader.line_num}")
                if any(row[i].strip() for i in blank):
                    raise InputValidationError(f"Unnamed GEOROC data column populated at line {reader.line_num}")
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
