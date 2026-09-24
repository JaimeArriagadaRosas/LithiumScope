from __future__ import annotations

import re
from urllib.parse import urljoin

import requests

from lithiumscope.tools.georoc_query_models import (
    Form,
    FormParser,
    Input,
    parse,
)


def norm(value: str) -> str:
    return re.sub(
        r"[^A-Z0-9]+",
        "",
        str(value).upper(),
    )


def default_payload(
    form: Form,
) -> list[tuple[str, str]]:
    payload: list[tuple[str, str]] = []
    for item in form.inputs:
        if not item.name:
            continue
        if item.kind in {
            "submit",
            "button",
            "image",
            "reset",
            "file",
        }:
            continue
        if (
            item.kind in {"checkbox", "radio"}
            and not item.checked
        ):
            continue
        payload.append(
            (item.name, item.value)
        )

    for select in form.selects:
        if not select.name:
            continue
        chosen = [
            option
            for option in select.options
            if option.selected
        ]
        if (
            not chosen
            and not select.multiple
            and select.options
        ):
            chosen = [select.options[0]]
        payload.extend(
            (select.name, option.value)
            for option in chosen
        )
    return payload


def replace_field(
    payload: list[tuple[str, str]],
    name: str,
    values: list[str],
) -> list[tuple[str, str]]:
    result = [
        (key, value)
        for key, value in payload
        if key != name
    ]
    result.extend(
        (name, value)
        for value in values
    )
    return result


def best_chemistry_form(
    forms: list[Form],
    chemistry: tuple[str, ...],
) -> Form:
    wanted = {
        norm(value)
        for value in chemistry
    }
    scored: list[
        tuple[int, Form]
    ] = []
    for form in forms:
        texts = {
            norm(option.text)
            for select in form.selects
            for option in select.options
        }
        scored.append(
            (
                len(wanted & texts),
                form,
            )
        )

    if not scored:
        raise RuntimeError(
            "GEOROC no entregó ningún "
            "formulario de consulta."
        )

    score, form = max(
        scored,
        key=lambda item: item[0],
    )
    if score < 3:
        raise RuntimeError(
            "No fue posible identificar el "
            "formulario químico de GEOROC."
        )
    return form


def select_chemistry(
    form: Form,
    payload: list[tuple[str, str]],
    chemistry: tuple[str, ...],
) -> list[tuple[str, str]]:
    wanted = {
        norm(value)
        for value in chemistry
    }
    for select in form.selects:
        values = [
            option.value
            for option in select.options
            if norm(option.text) in wanted
        ]
        if values:
            payload = replace_field(
                payload,
                select.name,
                values,
            )
    return payload


def _matching_inputs(
    items: list[Input],
    desired: tuple[str, ...],
) -> list[Input]:
    return [
        item
        for item in items
        if any(
            token
            in norm(
                item.nearby_text
                + " "
                + item.value
            )
            for token in desired
        )
    ]


def set_named_choices(
    form: Form,
    payload: list[tuple[str, str]],
    wanted_labels: tuple[str, ...],
) -> list[tuple[str, str]]:
    desired = tuple(
        norm(value)
        for value in wanted_labels
    )
    checkbox_names = {
        item.name
        for item in form.inputs
        if (
            item.kind
            in {"checkbox", "radio"}
            and item.name
        )
    }
    result = [
        (key, value)
        for key, value in payload
        if key not in checkbox_names
    ]

    groups: dict[
        str,
        list[Input],
    ] = {}
    for item in form.inputs:
        if (
            item.kind
            not in {"checkbox", "radio"}
            or not item.name
        ):
            continue
        groups.setdefault(
            item.name,
            [],
        ).append(item)

    for name, items in groups.items():
        matches = _matching_inputs(
            items,
            desired,
        )
        if matches:
            selected = (
                matches[:1]
                if any(
                    item.kind == "radio"
                    for item in items
                )
                else matches
            )
            result.extend(
                (name, item.value)
                for item in selected
            )
        else:
            result.extend(
                (name, item.value)
                for item in items
                if item.checked
            )
    return result


def add_submit(
    form: Form,
    payload: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    preferred = (
        "continue",
        "next",
        "query",
        "search",
        "download",
        "submit",
    )
    for item in form.inputs:
        if (
            item.kind
            not in {
                "submit",
                "button",
                "image",
            }
            or not item.name
        ):
            continue
        text = norm(
            item.value
            + " "
            + item.nearby_text
        )
        if any(
            norm(token) in text
            for token in preferred
        ):
            return payload + [
                (item.name, item.value)
            ]

    for item in form.inputs:
        if (
            item.kind == "submit"
            and item.name
        ):
            return payload + [
                (item.name, item.value)
            ]
    return payload


def submit_form(
    session: requests.Session,
    base_url: str,
    form: Form,
    payload: list[tuple[str, str]],
    timeout: float,
) -> requests.Response:
    url = urljoin(
        base_url,
        form.action or base_url,
    )
    if form.method == "post":
        response = session.post(
            url,
            data=payload,
            timeout=timeout,
        )
    else:
        response = session.get(
            url,
            params=payload,
            timeout=timeout,
        )
    response.raise_for_status()
    return response


def follow_link_by_text(
    session: requests.Session,
    base_url: str,
    parser: FormParser,
    tokens: tuple[str, ...],
    timeout: float,
) -> requests.Response | None:
    wanted = tuple(
        norm(token)
        for token in tokens
    )
    ranked: list[
        tuple[int, str]
    ] = []
    for href, text in parser.links:
        combined = norm(
            text + " " + href
        )
        score = sum(
            1
            for token in wanted
            if token in combined
        )
        if score:
            ranked.append(
                (score, href)
            )

    if not ranked:
        return None

    _, href = max(
        ranked,
        key=lambda item: item[0],
    )
    response = session.get(
        urljoin(
            base_url,
            href,
        ),
        timeout=timeout,
    )
    response.raise_for_status()
    return response


__all__ = [
    "Form",
    "FormParser",
    "add_submit",
    "best_chemistry_form",
    "default_payload",
    "follow_link_by_text",
    "norm",
    "parse",
    "replace_field",
    "select_chemistry",
    "set_named_choices",
    "submit_form",
]
