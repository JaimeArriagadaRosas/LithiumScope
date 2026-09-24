from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
import re
from urllib.parse import urljoin

import requests


def norm(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


@dataclass
class Option:
    value: str
    text: str
    selected: bool = False


@dataclass
class Select:
    name: str
    multiple: bool
    options: list[Option] = field(default_factory=list)


@dataclass
class Input:
    name: str
    value: str
    kind: str
    checked: bool
    nearby_text: str = ""


@dataclass
class Form:
    action: str
    method: str
    inputs: list[Input] = field(default_factory=list)
    selects: list[Select] = field(default_factory=list)


class FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms: list[Form] = []
        self.links: list[tuple[str, str]] = []
        self._form: Form | None = None
        self._select: Select | None = None
        self._option: Option | None = None
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []
        self._recent_input: Input | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        data = {str(k).lower(): str(v or "") for k, v in attrs}
        tag = tag.lower()
        if tag == "form":
            self._form = Form(
                action=data.get("action", ""),
                method=data.get("method", "get").lower(),
            )
            self.forms.append(self._form)
        elif tag == "select" and self._form is not None:
            self._select = Select(
                name=data.get("name", ""),
                multiple="multiple" in data,
            )
            self._form.selects.append(self._select)
        elif tag == "option" and self._select is not None:
            self._option = Option(
                value=data.get("value", ""),
                text="",
                selected="selected" in data,
            )
            self._select.options.append(self._option)
        elif tag == "input" and self._form is not None:
            item = Input(
                name=data.get("name", ""),
                value=data.get("value", ""),
                kind=data.get("type", "text").lower(),
                checked="checked" in data,
            )
            self._form.inputs.append(item)
            self._recent_input = item
        elif tag == "a":
            self._anchor_href = data.get("href", "")
            self._anchor_text = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "form":
            self._form = None
            self._recent_input = None
        elif tag == "select":
            self._select = None
        elif tag == "option":
            self._option = None
        elif tag == "a" and self._anchor_href is not None:
            self.links.append(
                (
                    self._anchor_href,
                    " ".join(self._anchor_text).strip(),
                )
            )
            self._anchor_href = None
            self._anchor_text = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._option is not None:
            self._option.text += (
                (" " if self._option.text else "") + text
            )
        if self._anchor_href is not None:
            self._anchor_text.append(text)
        if self._recent_input is not None:
            joined = (
                self._recent_input.nearby_text + " " + text
            ).strip()
            self._recent_input.nearby_text = joined[-240:]


def parse(html: str) -> FormParser:
    parser = FormParser()
    parser.feed(html)
    return parser


def default_payload(form: Form) -> list[tuple[str, str]]:
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
        payload.append((item.name, item.value))

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
    result.extend((name, value) for value in values)
    return result


def best_chemistry_form(
    forms: list[Form],
    chemistry: tuple[str, ...],
) -> Form:
    wanted = {norm(value) for value in chemistry}
    scored: list[tuple[int, Form]] = []
    for form in forms:
        texts = {
            norm(option.text)
            for select in form.selects
            for option in select.options
        }
        scored.append((len(wanted & texts), form))

    if not scored:
        raise RuntimeError(
            "GEOROC no entregó ningún formulario de consulta."
        )

    score, form = max(scored, key=lambda item: item[0])
    if score < 3:
        raise RuntimeError(
            "No fue posible identificar el formulario químico de GEOROC."
        )
    return form


def select_chemistry(
    form: Form,
    payload: list[tuple[str, str]],
    chemistry: tuple[str, ...],
) -> list[tuple[str, str]]:
    wanted = {norm(value) for value in chemistry}
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


def set_named_choices(
    form: Form,
    payload: list[tuple[str, str]],
    wanted_labels: tuple[str, ...],
) -> list[tuple[str, str]]:
    desired = tuple(norm(value) for value in wanted_labels)
    checkbox_names = {
        item.name
        for item in form.inputs
        if item.kind in {"checkbox", "radio"}
        and item.name
    }
    result = [
        (key, value)
        for key, value in payload
        if key not in checkbox_names
    ]

    groups: dict[str, list[Input]] = {}
    for item in form.inputs:
        if (
            item.kind not in {"checkbox", "radio"}
            or not item.name
        ):
            continue
        groups.setdefault(item.name, []).append(item)

    for name, items in groups.items():
        matches = [
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
            item.kind not in {"submit", "button", "image"}
            or not item.name
        ):
            continue
        text = norm(
            item.value + " " + item.nearby_text
        )
        if any(norm(token) in text for token in preferred):
            return payload + [(item.name, item.value)]

    for item in form.inputs:
        if item.kind == "submit" and item.name:
            return payload + [(item.name, item.value)]
    return payload


def submit_form(
    session: requests.Session,
    base_url: str,
    form: Form,
    payload: list[tuple[str, str]],
    timeout: float,
) -> requests.Response:
    url = urljoin(base_url, form.action or base_url)
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
    wanted = tuple(norm(token) for token in tokens)
    ranked: list[tuple[int, str]] = []
    for href, text in parser.links:
        combined = norm(text + " " + href)
        score = sum(
            1
            for token in wanted
            if token in combined
        )
        if score:
            ranked.append((score, href))

    if not ranked:
        return None

    _, href = max(ranked, key=lambda item: item[0])
    response = session.get(
        urljoin(base_url, href),
        timeout=timeout,
    )
    response.raise_for_status()
    return response
