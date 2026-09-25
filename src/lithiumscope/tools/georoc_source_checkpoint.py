from __future__ import annotations

from pathlib import Path

import requests

from lithiumscope.runtime.console_status import Spinner
from lithiumscope.tools.georoc_query_log import (
    BoundedRunLog,
)
from lithiumscope.tools.georoc_query_transfer import (
    stream_response_to_file,
)


def source_path_for(
    destination: Path,
) -> Path:
    return destination.with_name(
        destination.name + ".source"
    )


def has_source_checkpoint(
    destination: Path,
) -> bool:
    source = source_path_for(destination)
    return (
        source.is_file()
        and source.stat().st_size > 0
    )


def persist_response_source(
    response: requests.Response,
    destination: Path,
    spinner: Spinner,
    run_log: BoundedRunLog,
) -> Path:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    source = source_path_for(destination)
    temporary = destination.with_name(
        destination.name + ".download.part"
    )
    temporary.unlink(missing_ok=True)

    try:
        stream_response_to_file(
            response,
            temporary,
            spinner,
            run_log,
        )
        if temporary.stat().st_size <= 0:
            raise RuntimeError(
                "La descarga GEOROC quedó vacía."
            )
        temporary.replace(source)
        run_log.event(
            "source_checkpoint",
            "persistido",
            path=source,
            bytes=source.stat().st_size,
        )
        return source
    except BaseException as exc:
        temporary.unlink(missing_ok=True)
        run_log.event(
            "source_checkpoint",
            "error",
            error_type=type(exc).__name__,
            detail=str(exc),
        )
        raise
