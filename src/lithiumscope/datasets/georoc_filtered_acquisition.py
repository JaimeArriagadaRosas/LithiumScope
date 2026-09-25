from __future__ import annotations

from pathlib import Path

import requests

from lithiumscope.core.paths import DATA_DIR, LOGS_DIR
from lithiumscope.datasets.georoc_query_contract import (
    CHEMISTRY,
    GEOROC_FILTERED_NAME,
    GEOROC_QUERY_URL,
    acquisition_contract,
    write_acquisition_contract,
)
from lithiumscope.datasets.georoc_query_flow import (
    advance_query,
    compile_file_form,
    initial_query,
)
from lithiumscope.runtime.console_status import Spinner
from lithiumscope.tools.georoc_query_diagnostics import (
    diagnose_query_exception,
    format_query_failure,
)
from lithiumscope.tools.georoc_dataset_health import (
    assess_dataset_health_file,
)
from lithiumscope.tools.georoc_query_export import (
    find_download_link,
    looks_downloadable,
    materialize_download,
    validate_export,
)
from lithiumscope.tools.georoc_query_log import BoundedRunLog
from lithiumscope.tools.georoc_query_models import parse
from lithiumscope.tools.georoc_query_transfer import (
    request_compiled_export,
)
from lithiumscope.tools.georoc_query_payload import (
    best_chemistry_form,
    default_payload,
    select_chemistry,
)


def _save_debug(
    step: int,
    response: requests.Response,
) -> Path:
    directory = (
        LOGS_DIR
        / "lab"
        / "georoc_query"
    )
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = directory / f"step_{step:02d}.html"
    path.write_text(
        response.text,
        encoding="utf-8",
        errors="replace",
    )
    return path


def _finish_download(
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


def acquire_filtered_georoc(
    destination: Path | None = None,
    *,
    timeout: float = 120.0,
    max_steps: int = 12,
) -> Path:
    destination = destination or (
        DATA_DIR
        / "raw"
        / "model_1"
        / "georoc"
        / GEOROC_FILTERED_NAME
    )
    if (
        destination.is_file()
        and destination.stat().st_size > 0
    ):
        validate_export(destination)
        return destination

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "LithiumScope/0.3 "
                "academic-laboratory"
            )
        }
    )
    spinner = Spinner(
        "Descargando GEOROC filtrado"
    ).start()
    run_log = BoundedRunLog(
        LOGS_DIR
        / "lab"
        / "georoc_query"
        / "run.log"
    )
    run_log.event(
        "acquisition",
        "inicio",
        destination=destination,
    )
    last_debug: Path | None = None
    try:
        try:
            response = initial_query(
                session,
                timeout,
                capture_initial=lambda page: _save_debug(
                    0,
                    page,
                ),
            )
        except Exception as exc:
            failure = diagnose_query_exception(
                exc,
                filename="error_initial.html",
            )
            raise RuntimeError(
                "Falló la consulta inicial de GEOROC: "
                + format_query_failure(failure)
            ) from exc

        for step in range(
            1,
            max_steps + 1,
        ):
            spinner.update(
                "GEOROC: "
                f"consulta {step}/{max_steps}"
            )

            if looks_downloadable(response):
                return _finish_download(
                    response,
                    destination,
                    spinner,
                    run_log,
                )

            last_debug = _save_debug(
                step,
                response,
            )
            parser = parse(response.text)
            run_log.event(
                "step",
                "respuesta",
                step=step,
                status=response.status_code,
                url=response.url,
                content_type=response.headers.get(
                    "content-type",
                    "",
                ),
            )
            downloadable = find_download_link(
                session,
                response,
                parser,
                timeout,
                run_log,
            )
            if downloadable is not None:
                run_log.event(
                    "download_link",
                    "detectado",
                    url=downloadable.url,
                )
            if downloadable is not None:
                return _finish_download(
                    downloadable,
                    destination,
                    spinner,
                    run_log,
                )

            compile_form = compile_file_form(
                parser.forms
            )
            if compile_form is not None:
                response = request_compiled_export(
                    session,
                    response.url,
                    compile_form,
                    default_payload(compile_form),
                    min(float(timeout), 30.0),
                    spinner,
                    run_log,
                )
                continue

            response = advance_query(
                session,
                response,
                timeout,
            )

        raise RuntimeError(
            "GEOROC no produjo una "
            "exportación filtrada."
        )
    except KeyboardInterrupt:
        run_log.event(
            "acquisition",
            "cancelada por usuario",
        )
        spinner.fail(
            "Descarga GEOROC cancelada"
        )
        raise
    except Exception as exc:
        run_log.event(
            "acquisition",
            "error",
            error_type=type(exc).__name__,
            detail=str(exc),
            evidence=last_debug,
        )
        spinner.fail(
            "Falló la descarga de GEOROC filtrado"
        )
        evidence = (
            f" Evidencia: {last_debug}."
            if last_debug is not None
            else ""
        )
        detail = str(exc).strip()
        detail_text = (
            f" Detalle: {detail}."
            if detail
            else ""
        )
        raise RuntimeError(
            "No se pudo completar la consulta "
            "filtrada de GEOROC."
            + evidence
            + detail_text
            + " Log: "
            + str(run_log.path)
            + ". No se descargó el paquete "
            "precompilado masivo."
        ) from exc


# Compatibility exports for focused tests.
_parse = parse
_default_payload = default_payload


def _best_chemistry_form(forms):
    return best_chemistry_form(
        forms,
        CHEMISTRY,
    )


def _select_chemistry(
    form,
    payload,
):
    return select_chemistry(
        form,
        payload,
        CHEMISTRY,
    )


__all__ = [
    "CHEMISTRY",
    "GEOROC_FILTERED_NAME",
    "GEOROC_QUERY_URL",
    "acquire_filtered_georoc",
    "acquisition_contract",
    "write_acquisition_contract",
]
