from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import requests

from lithiumscope.runtime.console_status import Spinner
from lithiumscope.tools.georoc_postprocess import (
    process_georoc_text_export,
)
from lithiumscope.tools.georoc_query_html import (
    norm,
)
from lithiumscope.tools.georoc_query_log import BoundedRunLog
from lithiumscope.tools.georoc_query_transfer import (
    stream_response_to_file,
)


def looks_downloadable(
    response: requests.Response,
) -> bool:
    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()
    disposition = response.headers.get(
        "content-disposition",
        "",
    ).lower()
    return (
        "text/csv" in content_type
        or "application/csv" in content_type
        or "spreadsheet" in content_type
        or "excel" in content_type
        or "application/zip" in content_type
        or "attachment" in disposition
    )


def materialize_download(
    response: requests.Response,
    destination: Path,
    *,
    spinner: Spinner | None = None,
    run_log: BoundedRunLog | None = None,
) -> Path:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    raw_path = destination.with_name(
        destination.name + ".download.part"
    )
    source_path = destination.with_name(
        destination.name + ".source"
    )
    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()
    disposition = response.headers.get(
        "content-disposition",
        "",
    ).lower()

    raw_path.unlink(missing_ok=True)

    try:
        if spinner is not None and run_log is not None:
            stream_response_to_file(
                response,
                raw_path,
                spinner,
                run_log,
            )
        else:
            raw_path.write_bytes(
                response.content
            )

        if run_log is not None:
            run_log.event(
                "postprocess",
                "inicio",
                raw_bytes=raw_path.stat().st_size,
                content_type=content_type,
                disposition=disposition,
            )

        if (
            "zip" in content_type
            or ".zip" in disposition
        ):
            with zipfile.ZipFile(raw_path) as archive:
                candidates = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith(
                        (".csv", ".txt")
                    )
                ]
                if not candidates:
                    raise RuntimeError(
                        "La exportación GEOROC ZIP no contiene "
                        "un CSV/TXT compatible."
                    )
                extracted = raw_path.with_name(
                    raw_path.name + ".table"
                )
                extracted.write_bytes(
                    archive.read(candidates[0])
                )
                try:
                    result = process_georoc_text_export(
                        extracted,
                        destination,
                    )
                finally:
                    extracted.unlink(
                        missing_ok=True
                    )
        elif (
            "excel" in content_type
            or ".xlsx" in disposition
            or ".xls" in disposition
        ):
            raise RuntimeError(
                "La ruta automática GEOROC esperaba una "
                "exportación de texto CSV/TXT, pero recibió Excel."
            )
        else:
            result = process_georoc_text_export(
                raw_path,
                destination,
            )

        raw_path.replace(source_path)

        if run_log is not None:
            run_log.event(
                "postprocess",
                "completado",
                encoding=result.encoding,
                rows_before_filter=(
                    result.rows_before_filter
                ),
                rows_after_filter=(
                    result.rows_after_filter
                ),
                rows_removed_by_material=(
                    result.rows_removed_by_material
                ),
                material_column=(
                    result.material_column
                ),
                source_copy=source_path,
            )
        return result.path
    except BaseException as exc:
        if raw_path.exists():
            try:
                raw_path.replace(source_path)
            except OSError:
                pass
        if run_log is not None:
            run_log.event(
                "postprocess",
                "error",
                error_type=type(exc).__name__,
                detail=str(exc),
                source_copy=(
                    source_path
                    if source_path.exists()
                    else None
                ),
            )
        raise


def validate_export(
    path: Path,
) -> None:
    try:
        frame = pd.read_csv(path, nrows=20)
    except Exception as exc:
        raise RuntimeError(
            "La exportación GEOROC no pudo leerse "
            f"como CSV: {exc}"
        ) from exc

    normalized = {
        norm(column)
        for column in frame.columns
    }
    required_groups = {
        "LI": ("LI", "LIPPM"),
        "LONGITUDE": (
            "LONGITUDE",
            "LONGITUDEMIN",
        ),
        "LATITUDE": (
            "LATITUDE",
            "LATITUDEMIN",
        ),
    }
    missing = [
        label
        for label, aliases in required_groups.items()
        if not any(
            any(alias in column for alias in aliases)
            for column in normalized
        )
    ]
    if missing:
        raise RuntimeError(
            "La exportación obtenida no corresponde "
            "al contrato LithiumScope. Faltan: "
            + ", ".join(missing)
        )
