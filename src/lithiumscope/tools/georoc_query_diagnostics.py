from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests

from lithiumscope.core.paths import LOGS_DIR


@dataclass(frozen=True)
class QueryFailure:
    kind: str
    message: str
    url: str | None
    status_code: int | None
    evidence_path: Path | None


def _request_url(
    exc: BaseException,
) -> str | None:
    request = getattr(exc, "request", None)
    url = getattr(request, "url", None)
    if url:
        return str(url)

    response = getattr(exc, "response", None)
    response_url = getattr(response, "url", None)
    if response_url:
        return str(response_url)
    return None


def _save_response_body(
    response: requests.Response | None,
    filename: str,
) -> Path | None:
    if response is None:
        return None

    directory = (
        LOGS_DIR
        / "lab"
        / "georoc_query"
    )
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = directory / filename
    path.write_text(
        response.text,
        encoding="utf-8",
        errors="replace",
    )
    return path


def diagnose_query_exception(
    exc: BaseException,
    *,
    filename: str = "error_initial.html",
) -> QueryFailure:
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(
            response,
            "status_code",
            None,
        )
        evidence = _save_response_body(
            response,
            filename,
        )
    else:
        status_code = None
        evidence = None

    return QueryFailure(
        kind=type(exc).__name__,
        message=str(exc),
        url=_request_url(exc),
        status_code=status_code,
        evidence_path=evidence,
    )


def format_query_failure(
    failure: QueryFailure,
) -> str:
    parts = [
        f"tipo={failure.kind}",
    ]
    if failure.status_code is not None:
        parts.append(
            f"HTTP={failure.status_code}"
        )
    if failure.url:
        parts.append(
            f"URL={failure.url}"
        )
    if failure.message:
        parts.append(
            f"detalle={failure.message}"
        )
    if failure.evidence_path:
        parts.append(
            f"evidencia={failure.evidence_path}"
        )
    return " | ".join(parts)
