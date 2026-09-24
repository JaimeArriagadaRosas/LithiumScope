from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import urljoin
import zipfile

import pandas as pd
import requests

from lithiumscope.core.logger import get_logger
from lithiumscope.core.paths import DATA_DIR, LOGS_DIR
from lithiumscope.runtime.console_status import Spinner

logger = get_logger("datasets.georoc_filtered_acquisition")

GEOROC_QUERY_URL = "https://georoc.eu/georoc/Chemistry.asp"
GEOROC_FILTERED_NAME = "GEOROC_Andean_Arc_LithiumScope.csv"

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

_CORE_OUTPUT_MARKERS = (
    "LI",
    "LONGITUDE",
    "LATITUDE",
)


def _norm(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


@dataclass
class _Option:
    value: str
    text: str
    selected: bool = False


@dataclass
class _Select:
    name: str
    multiple: bool
    options: list[_Option] = field(default_factory=list)


@dataclass
class _Input:
    name: str
    value: str
    kind: str
    checked: bool
    nearby_text: str = ""


@dataclass
class _Form:
    action: str
    method: str
    inputs: list[_Input] = field(default_factory=list)
    selects: list[_Select] = field(default_factory=list)


class _FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms: list[_Form] = []
        self.links: list[tuple[str, str]] = []
        self._form: _Form | None = None
        self._select: _Select | None = None
        self._option: _Option | None = None
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []
        self._recent_input: _Input | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        data = {str(k).lower(): str(v or "") for k, v in attrs}
        tag = tag.lower()
        if tag == "form":
            self._form = _Form(
                action=data.get("action", ""),
                method=data.get("method", "get").lower(),
            )
            self.forms.append(self._form)
        elif tag == "select" and self._form is not None:
            self._select = _Select(
                name=data.get("name", ""),
                multiple="multiple" in data,
            )
            self._form.selects.append(self._select)
        elif tag == "option" and self._select is not None:
            self._option = _Option(
                value=data.get("value", ""),
                text="",
                selected="selected" in data,
            )
            self._select.options.append(self._option)
        elif tag == "input" and self._form is not None:
            item = _Input(
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
                (self._anchor_href, " ".join(self._anchor_text).strip())
            )
            self._anchor_href = None
            self._anchor_text = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._option is not None:
            self._option.text += (" " if self._option.text else "") + text
        if self._anchor_href is not None:
            self._anchor_text.append(text)
        if self._recent_input is not None:
            joined = (
                self._recent_input.nearby_text + " " + text
            ).strip()
            self._recent_input.nearby_text = joined[-240:]


def _parse(html: str) -> _FormParser:
    parser = _FormParser()
    parser.feed(html)
    return parser


def _default_payload(form: _Form) -> list[tuple[str, str]]:
    payload: list[tuple[str, str]] = []
    for item in form.inputs:
        if not item.name:
            continue
        if item.kind in {"submit", "button", "image", "reset", "file"}:
            continue
        if item.kind in {"checkbox", "radio"} and not item.checked:
            continue
        payload.append((item.name, item.value))
    for select in form.selects:
        if not select.name:
            continue
        chosen = [opt for opt in select.options if opt.selected]
        if not chosen and not select.multiple and select.options:
            chosen = [select.options[0]]
        for option in chosen:
            payload.append((select.name, option.value))
    return payload


def _replace_field(
    payload: list[tuple[str, str]],
    name: str,
    values: list[str],
) -> list[tuple[str, str]]:
    result = [(key, value) for key, value in payload if key != name]
    result.extend((name, value) for value in values)
    return result


def _best_chemistry_form(forms: list[_Form]) -> _Form:
    scored: list[tuple[int, _Form]] = []
    wanted = {_norm(value) for value in CHEMISTRY}
    for form in forms:
        texts = {
            _norm(option.text)
            for select in form.selects
            for option in select.options
        }
        score = len(wanted & texts)
        scored.append((score, form))
    if not scored:
        raise RuntimeError("GEOROC no entregó ningún formulario de consulta.")
    score, form = max(scored, key=lambda item: item[0])
    if score < 3:
        raise RuntimeError(
            "No fue posible identificar el formulario químico de GEOROC."
        )
    return form


def _select_chemistry(
    form: _Form,
    payload: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    wanted = {_norm(value) for value in CHEMISTRY}
    for select in form.selects:
        values = [
            option.value
            for option in select.options
            if _norm(option.text) in wanted
        ]
        if values:
            payload = _replace_field(payload, select.name, values)
    return payload


def _set_named_choices(
    form: _Form,
    payload: list[tuple[str, str]],
    wanted_labels: tuple[str, ...],
) -> list[tuple[str, str]]:
    desired = tuple(_norm(value) for value in wanted_labels)
    result = [
        (key, value)
        for key, value in payload
        if not any(
            item.name == key
            and item.kind in {"checkbox", "radio"}
            for item in form.inputs
        )
    ]
    groups: dict[str, list[_Input]] = {}
    for item in form.inputs:
        if item.kind not in {"checkbox", "radio"} or not item.name:
            continue
        groups.setdefault(item.name, []).append(item)

    for name, items in groups.items():
        matches = [
            item
            for item in items
            if any(
                token in _norm(
                    item.nearby_text + " " + item.value
                )
                for token in desired
            )
        ]
        if matches:
            # For radio groups choose the first requested match; checkboxes can stack.
            selected = matches[:1] if any(i.kind == "radio" for i in items) else matches
            result.extend((name, item.value) for item in selected)
        else:
            defaults = [item for item in items if item.checked]
            result.extend((name, item.value) for item in defaults)
    return result


def _add_submit(
    form: _Form,
    payload: list[tuple[str, str]],
    preferred: tuple[str, ...] = ("continue", "next", "query", "search", "download"),
) -> list[tuple[str, str]]:
    for item in form.inputs:
        if item.kind not in {"submit", "button", "image"} or not item.name:
            continue
        text = _norm(item.value + " " + item.nearby_text)
        if any(_norm(token) in text for token in preferred):
            return payload + [(item.name, item.value)]
    for item in form.inputs:
        if item.kind == "submit" and item.name:
            return payload + [(item.name, item.value)]
    return payload


def _submit(
    session: requests.Session,
    base_url: str,
    form: _Form,
    payload: list[tuple[str, str]],
    timeout: float,
) -> requests.Response:
    url = urljoin(base_url, form.action or base_url)
    if form.method == "post":
        response = session.post(url, data=payload, timeout=timeout)
    else:
        response = session.get(url, params=payload, timeout=timeout)
    response.raise_for_status()
    return response


def _save_debug(step: int, response: requests.Response) -> Path:
    directory = LOGS_DIR / "lab" / "georoc_query"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{step:02d}.html"
    path.write_text(response.text, encoding="utf-8", errors="replace")
    return path


def _looks_downloadable(response: requests.Response) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    disposition = response.headers.get("content-disposition", "").lower()
    return (
        "text/csv" in content_type
        or "application/csv" in content_type
        or "spreadsheet" in content_type
        or "excel" in content_type
        or "application/zip" in content_type
        or "attachment" in disposition
    )


def _materialize_download(
    response: requests.Response,
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    content_type = response.headers.get("content-type", "").lower()
    disposition = response.headers.get("content-disposition", "").lower()

    if "zip" in content_type or ".zip" in disposition:
        with zipfile.ZipFile(BytesIO(response.content)) as archive:
            candidates = [
                name for name in archive.namelist()
                if name.lower().endswith((".csv", ".txt", ".xlsx", ".xls"))
            ]
            if not candidates:
                raise RuntimeError("La exportación GEOROC ZIP no contiene una tabla compatible.")
            raw = archive.read(candidates[0])
            suffix = Path(candidates[0]).suffix.lower()
    else:
        raw = response.content
        suffix = ".xlsx" if ("excel" in content_type or ".xlsx" in disposition) else ".csv"

    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(BytesIO(raw))
        frame.to_csv(destination, index=False)
    else:
        # Preserve the server text first, then normalize delimiters through pandas.
        temporary = destination.with_suffix(".download")
        temporary.write_bytes(raw)
        try:
            frame = pd.read_csv(
                temporary,
                sep=None,
                engine="python",
            )
        finally:
            temporary.unlink(missing_ok=True)
        frame.to_csv(destination, index=False)
    return destination


def _validate_export(path: Path) -> None:
    try:
        frame = pd.read_csv(path, nrows=20)
    except Exception as exc:
        raise RuntimeError(
            f"La exportación GEOROC no pudo leerse como CSV: {exc}"
        ) from exc

    normalized = {_norm(column) for column in frame.columns}
    missing = [
        marker
        for marker in _CORE_OUTPUT_MARKERS
        if not any(marker in column for column in normalized)
    ]
    if missing:
        raise RuntimeError(
            "La exportación obtenida no parece corresponder al contrato "
            f"LithiumScope. Faltan marcadores: {', '.join(missing)}."
        )


def _download_link(
    session: requests.Session,
    response: requests.Response,
    parser: _FormParser,
    timeout: float,
) -> requests.Response | None:
    ranked: list[tuple[int, str]] = []
    for href, text in parser.links:
        joined = (href + " " + text).lower()
        score = 0
        if "download" in joined:
            score += 5
        if any(ext in joined for ext in (".csv", ".txt", ".xlsx", ".xls", ".zip")):
            score += 4
        if "data" in joined:
            score += 1
        if score:
            ranked.append((score, href))
    for _, href in sorted(ranked, reverse=True):
        candidate = session.get(
            urljoin(response.url, href),
            timeout=timeout,
        )
        candidate.raise_for_status()
        if _looks_downloadable(candidate):
            return candidate
    return None


def _advance_form(
    session: requests.Session,
    response: requests.Response,
    parser: _FormParser,
    timeout: float,
) -> requests.Response:
    # Geological setting page: select only ANDEAN ARC when offered.
    for form in parser.forms:
        for select in form.selects:
            matches = [
                option.value
                for option in select.options
                if "ANDEANARC" in _norm(option.text)
            ]
            if matches:
                payload = _default_payload(form)
                payload = _replace_field(payload, select.name, matches[:1])
                payload = _add_submit(form, payload)
                return _submit(session, response.url, form, payload, timeout)

    # Result/output pages: prefer CSV/text and one-row/sample controls by label.
    for form in parser.forms:
        payload = _default_payload(form)
        payload = _set_named_choices(
            form,
            payload,
            (
                "CSV",
                "TEXT FILE",
                "ONE ROW PER SAMPLE",
                "STANDARD OUTPUT",
                "WHOLE ROCK",
                "COMPILED",
            ),
        )
        submitted = _add_submit(form, payload)
        if submitted != payload or form.inputs:
            return _submit(session, response.url, form, submitted, timeout)

    raise RuntimeError("GEOROC no entregó un formulario reconocible para continuar.")


def acquire_filtered_georoc(
    destination: Path | None = None,
    *,
    timeout: float = 120.0,
    max_steps: int = 10,
) -> Path:
    destination = destination or (
        DATA_DIR / "raw" / "model_1" / "georoc" / GEOROC_FILTERED_NAME
    )
    if destination.is_file() and destination.stat().st_size > 0:
        _validate_export(destination)
        return destination

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "LithiumScope/0.3 laboratory data acquisition "
                "(academic reproducibility)"
            )
        }
    )
    spinner = Spinner(
        "Consultando GEOROC para extracción filtrada Andean Arc"
    ).start()

    try:
        response = session.get(GEOROC_QUERY_URL, timeout=timeout)
        response.raise_for_status()
        parser = _parse(response.text)
        form = _best_chemistry_form(parser.forms)
        payload = _default_payload(form)
        payload = _select_chemistry(form, payload)
        payload = _set_named_choices(
            form,
            payload,
            (
                "COMPILED",
                "ALL ROCK TYPES",
                "WHOLE ROCK",
            ),
        )
        payload = _add_submit(form, payload)
        response = _submit(
            session,
            response.url,
            form,
            payload,
            timeout,
        )

        for step in range(1, max_steps + 1):
            spinner.update(
                f"GEOROC filtrado: resolviendo etapa remota {step}/{max_steps}"
            )
            if _looks_downloadable(response):
                path = _materialize_download(response, destination)
                _validate_export(path)
                spinner.succeed(
                    f"GEOROC filtrado disponible: {path.name}"
                )
                return path

            debug_path = _save_debug(step, response)
            parser = _parse(response.text)
            downloadable = _download_link(
                session,
                response,
                parser,
                timeout,
            )
            if downloadable is not None:
                path = _materialize_download(downloadable, destination)
                _validate_export(path)
                spinner.succeed(
                    f"GEOROC filtrado disponible: {path.name}"
                )
                return path

            try:
                response = _advance_form(
                    session,
                    response,
                    parser,
                    timeout,
                )
            except Exception as exc:
                raise RuntimeError(
                    "La interfaz pública de GEOROC cambió o no permitió "
                    "completar la exportación filtrada. "
                    f"Última evidencia: {debug_path}. "
                    "Por seguridad LithiumScope NO descargará los 22 GiB "
                    "precompilados como fallback."
                ) from exc

        raise RuntimeError(
            "GEOROC no produjo una descarga filtrada dentro del número "
            "máximo de etapas. No se ejecutó ningún fallback masivo."
        )
    except Exception:
        spinner.fail("No fue posible adquirir GEOROC filtrado")
        raise


def acquisition_contract() -> dict:
    return {
        "query_url": GEOROC_QUERY_URL,
        "scope": "ANDEAN ARC",
        "material": "WHOLE ROCK",
        "chemistry": list(CHEMISTRY),
        "destination_name": GEOROC_FILTERED_NAME,
        "massive_precompiled_fallback": False,
    }


def write_acquisition_contract(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(acquisition_contract(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
