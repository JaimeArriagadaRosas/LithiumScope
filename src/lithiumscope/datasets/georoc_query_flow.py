from __future__ import annotations

from collections.abc import Callable

import requests

from lithiumscope.datasets.georoc_query_contract import (
    CHEMISTRY,
    GEOROC_QUERY_URL,
)
from lithiumscope.tools.georoc_query_html import (
    Form,
    follow_link_by_text,
    norm,
    parse,
    submit_form,
)
from lithiumscope.tools.georoc_query_payload import (
    add_submit,
    best_chemistry_form,
    default_payload,
    replace_field,
    select_chemistry,
    select_submit_by_label,
    set_named_choices,
)


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
        return add_submit(
            form,
            chosen,
        )
    return None


def _is_chem_location_form(
    form: Form,
) -> bool:
    action = norm(form.action)
    has_submit = any(
        item.kind
        in {"submit", "button", "image"}
        for item in form.inputs
    )
    return (
        form.method == "post"
        and "CHEMLOCASP" in action
        and has_submit
    )


def initial_query(
    session: requests.Session,
    timeout: float,
    capture_initial: Callable[
        [requests.Response],
        None,
    ]
    | None = None,
) -> requests.Response:
    response = session.get(
        GEOROC_QUERY_URL,
        timeout=timeout,
    )
    response.raise_for_status()
    if capture_initial is not None:
        capture_initial(response)
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
    payload = select_submit_by_label(
        form,
        payload,
        ("CONVERGENT MARGIN",),
    )
    return submit_form(
        session,
        response.url,
        form,
        payload,
        timeout,
    )


def advance_query(
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
        payload = _select_andean_form(
            form
        )
        if payload is not None:
            return submit_form(
                session,
                response.url,
                form,
                payload,
                timeout,
            )

    # GEOROC's current Chemistry results page continues
    # through POST ChemLoc.asp. Its visible Continue button
    # has no name attribute, so the hidden fields themselves
    # are the request payload.
    for form in parser.forms:
        if _is_chem_location_form(form):
            return submit_form(
                session,
                response.url,
                form,
                default_payload(form),
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
        "GEOROC no entregó un paso "
        "de consulta reconocible."
    )
