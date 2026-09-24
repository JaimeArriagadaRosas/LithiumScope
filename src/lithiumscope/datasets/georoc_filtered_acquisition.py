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
    initial_query,
)
from lithiumscope.runtime.console_status import Spinner
from lithiumscope.tools.georoc_query_export import (
    find_download_link,
    looks_downloadable,
    materialize_download,
    validate_export,
)
from lithiumscope.tools.georoc_query_html import (
    best_chemistry_form,
    default_payload,
    parse,
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
) -> Path:
    path = materialize_download(
        response,
        destination,
    )
    validate_export(path)
    spinner.succeed(
        "GEOROC filtrado descargado"
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

    last_debug: Path | None = None
    try:
        response = initial_query(
            session,
            timeout,
        )

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
                )

            last_debug = _save_debug(
                step,
                response,
            )
            parser = parse(response.text)
            downloadable = find_download_link(
                session,
                response,
                parser,
                timeout,
            )
            if downloadable is not None:
                return _finish_download(
                    downloadable,
                    destination,
                    spinner,
                )

            response = advance_query(
                session,
                response,
                timeout,
            )

        raise RuntimeError(
            "GEOROC no produjo una "
            "exportación filtrada."
        )
    except Exception as exc:
        spinner.fail(
            "Falló la descarga de GEOROC filtrado"
        )
        evidence = (
            f" Evidencia: {last_debug}."
            if last_debug is not None
            else ""
        )
        raise RuntimeError(
            "No se pudo completar la consulta "
            "filtrada de GEOROC."
            + evidence
            + " No se descargó el paquete "
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
