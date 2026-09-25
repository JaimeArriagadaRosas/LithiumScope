from __future__ import annotations

from pathlib import Path

import requests

from lithiumscope.runtime.console_status import Spinner
from lithiumscope.tools.georoc_dataset_health import (
    assess_dataset_health_file,
)
from lithiumscope.tools.georoc_query_export import (
    materialize_download,
    validate_export,
)
from lithiumscope.tools.georoc_postprocess import (
    process_georoc_text_export,
)
from lithiumscope.tools.georoc_query_log import (
    BoundedRunLog,
)
from lithiumscope.tools.georoc_schema_profile import (
    log_georoc_schema_profile,
)
from lithiumscope.tools.georoc_source_checkpoint import (
    source_path_for,
)


def finish_download(
    response: requests.Response,
    destination: Path,
    spinner: Spinner,
    run_log: BoundedRunLog,
) -> Path:
    path = materialize_download(
        response,
        destination,
        spinner=spinner,
        run_log=run_log,
    )
    validate_export(path)
    health = assess_dataset_health_file(path)
    run_log.event(
        "health_check",
        "salud post-descarga OK",
        **health.as_log_fields(),
    )
    spinner.succeed(
        "GEOROC filtrado descargado y validado"
    )
    return path



def finish_source_checkpoint(
    destination: Path,
    spinner: Spinner,
    run_log: BoundedRunLog,
) -> Path:
    source = source_path_for(destination)
    if (
        not source.is_file()
        or source.stat().st_size <= 0
    ):
        raise RuntimeError(
            "No existe un checkpoint GEOROC reutilizable."
        )

    spinner.update(
        "GEOROC → procesando fuente ya descargada"
    )
    run_log.event(
        "resume",
        "procesando checkpoint local",
        source=source,
        bytes=source.stat().st_size,
    )
    result = process_georoc_text_export(
        source,
        destination,
        profile_callback=lambda profile: (
            log_georoc_schema_profile(
                run_log,
                profile,
            )
        ),
    )
    validate_export(result.path)
    health = assess_dataset_health_file(
        result.path
    )
    run_log.event(
        "postprocess",
        "completado desde checkpoint",
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
    )
    run_log.event(
        "health_check",
        "salud post-descarga OK",
        **health.as_log_fields(),
    )
    spinner.succeed(
        "GEOROC recuperado, filtrado y validado"
    )
    return result.path
