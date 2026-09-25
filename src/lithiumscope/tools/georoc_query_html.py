from __future__ import annotations

import re
from urllib.parse import urljoin

import requests

from lithiumscope.tools.georoc_query_models import (
    Form,
    FormParser,
    parse,
)


def norm(value: str) -> str:
    return re.sub(
        r"[^A-Z0-9]+",
        "",
        str(value).upper(),
    )


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
        if str(href).strip().lower().startswith(
            "javascript:"
        ):
            continue
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
    "follow_link_by_text",
    "norm",
    "parse",
    "submit_form",
]
