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
from lithiumscope.tools.georoc_query_log import (
    BoundedRunLog,
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
