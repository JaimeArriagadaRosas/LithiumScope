from __future__ import annotations

import json
from pathlib import Path

import requests

from lithiumscope.core.paths import DATA_DIR, LOGS_DIR
from lithiumscope.runtime.console_status import Spinner
from lithiumscope.tools.georoc_query_export import (
    find_download_link,
    looks_downloadable,
    materialize_download,
    validate_export,
)
from lithiumscope.tools.georoc_query_html import (
    Form,
    add_submit,
    best_chemistry_form,
    default_payload,
    follow_link_by_text,
    norm,
    parse,
    replace_field,
    select_chemistry,
    set_named_choices,
    submit_form,
)

GEOROC_QUERY_URL = (
    "https://georoc.eu/georoc/Chemistry.asp"
)
GEOROC_FILTERED_NAME = (
    "GEOROC_Andean_Arc_LithiumScope.csv"
)

# Only chemistry currently used by Model 1.
CHEMISTRY = (
    "LI",
    "SIO2",
    "TIO2",
    "AL2O3",
    "FE2O3",
    "FE2O3T",
    "FEOT",
    "MNO",
    "MGO",
    "CAO",
    "NA2O",
    "K2O",
    "P2O5",
    "TH",
    "U",
    "RB",
    "CS",
    "NB",
    "TA",
    "PB",
    "BA",
    "SR",
    "ZR",
    "V",
    "HF",
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


def _select_andean_form(
    form: Form,
) -> list[tuple[str, str]] | None:
    payload = default_payload(form)

    for select in form.selects:
        matches = [
            option.value
            for option in select.options
            if "ANDEANARC" in norm(option.text)
        ]
        if matches:
            return add_submit(
                form,
                replace_field(
                    payload,
                    select.name,
                    matches[:1],
                ),
            )

    chosen = set_named_choices(
        form,
        payload,
        (
            "ANDEAN ARC",
            "CONVERGENT MARGIN",
        ),
    )
    if chosen != payload:
        return add_submit(form, chosen)
    return None


def _advance(
    session: requests.Session,
    response: requests.Response,
    timeout: float,
) -> requests.Response:
    parser = parse(response.text)

    direct = follow_link_by_text(
        session,
        response.url,
        parser,
        ("ANDEAN ARC",),
        timeout,
    )
    if direct is not None:
        return direct

    for form in parser.forms:
        payload = _select_andean_form(form)
        if payload is not None:
            return submit_form(
                session,
                response.url,
                form,
                payload,
                timeout,
            )

    convergent = follow_link_by_text(
        session,
        response.url,
        parser,
        (
            "CONVERGENT MARGIN",
            "CONVERGENT MARGINS",
        ),
        timeout,
    )
    if convergent is not None:
        return convergent

    for form in parser.forms:
        payload = default_payload(form)
        selected = set_named_choices(
            form,
            payload,
            (
                "WHOLE ROCK",
                "COMPILED",
                "ONE ROW PER SAMPLE",
                "CSV",
                "TEXT FILE",
                "STANDARD OUTPUT",
            ),
        )
        submitted = add_submit(
            form,
            selected,
        )
        if submitted != payload:
            return submit_form(
                session,
                response.url,
                form,
                submitted,
                timeout,
            )

    continuation = follow_link_by_text(
        session,
        response.url,
        parser,
        (
            "CONTINUE",
            "SAMPLE CRITERIA",
            "OUTPUT",
            "DOWNLOAD",
        ),
        timeout,
    )
    if continuation is not None:
        return continuation

    raise RuntimeError(
        "GEOROC no entregó un paso de consulta "
        "reconocible."
    )


def _initial_query(
    session: requests.Session,
    timeout: float,
) -> requests.Response:
    response = session.get(
        GEOROC_QUERY_URL,
        timeout=timeout,
    )
    response.raise_for_status()
    parser = parse(response.text)
    form = best_chemistry_form(
        parser.forms,
        CHEMISTRY,
    )
    payload = default_payload(form)
    payload = select_chemistry(
        form,
        payload,
        CHEMISTRY,
    )
    payload = set_named_choices(
        form,
        payload,
        (
            "COMPILED",
            "ALL ROCK TYPES",
            "WHOLE ROCK",
        ),
    )
    payload = add_submit(form, payload)
    return submit_form(
        session,
        response.url,
        form,
        payload,
        timeout,
    )


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
        response = _initial_query(
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
                path = materialize_download(
                    response,
                    destination,
                )
                validate_export(path)
                spinner.succeed(
                    "GEOROC filtrado descargado"
                )
                return path

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
                path = materialize_download(
                    downloadable,
                    destination,
                )
                validate_export(path)
                spinner.succeed(
                    "GEOROC filtrado descargado"
                )
                return path

            response = _advance(
                session,
                response,
                timeout,
            )

        raise RuntimeError(
            "GEOROC no produjo una exportación "
            "filtrada."
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


def acquisition_contract() -> dict:
    return {
        "query_url": GEOROC_QUERY_URL,
        "scope": "ANDEAN ARC",
        "material": "WHOLE ROCK",
        "chemistry": list(CHEMISTRY),
        "destination_name": (
            GEOROC_FILTERED_NAME
        ),
        "massive_precompiled_fallback": False,
    }


def write_acquisition_contract(
    path: Path,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            acquisition_contract(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


# Backwards-compatible aliases used by focused unit tests.
_parse = parse
_default_payload = default_payload


def _best_chemistry_form(
    forms,
):
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
