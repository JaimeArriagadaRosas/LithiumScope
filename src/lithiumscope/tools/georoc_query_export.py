from __future__ import annotations

from pathlib import Path
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
from lithiumscope.tools.georoc_schema_profile import (
    log_georoc_schema_profile,
)
from lithiumscope.tools.georoc_source_checkpoint import (
    persist_response_source,
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
    if spinner is None or run_log is None:
        raise RuntimeError(
            "La materialización GEOROC requiere "
            "estado de consola y log de ejecución."
        )

    source_path = persist_response_source(
        response,
        destination,
        spinner,
        run_log,
    )
    run_log.event(
        "postprocess",
        "inicio desde checkpoint",
        source_copy=source_path,
        raw_bytes=source_path.stat().st_size,
        content_type=response.headers.get(
            "content-type",
            "",
        ),
        disposition=response.headers.get(
            "content-disposition",
            "",
        ),
    )

    result = process_georoc_text_export(
        source_path,
        destination,
        profile_callback=lambda profile: (
            log_georoc_schema_profile(
                run_log,
                profile,
            )
        ),
    )
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
        whole_rock_rows=result.whole_rock_rows,
        volcanic_glass_rows=(
            result.volcanic_glass_rows
        ),
        unknown_rows=result.unknown_rows,
        missing_rows=result.missing_rows,
        source_copy=source_path,
    )
    return result.path


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
