from __future__ import annotations

from urllib.parse import urljoin

import requests

from lithiumscope.tools.georoc_query_download_links import (
    resolve_download_href,
)
from lithiumscope.tools.georoc_query_export import (
    looks_downloadable,
)
from lithiumscope.tools.georoc_query_log import (
    BoundedRunLog,
)
from lithiumscope.tools.georoc_query_models import (
    FormParser,
)


_EXTENSIONS = (
    ".csv",
    ".txt",
    ".xlsx",
    ".xls",
    ".zip",
)


def find_download_link(
    session: requests.Session,
    response: requests.Response,
    parser: FormParser,
    timeout: float,
    run_log: BoundedRunLog | None = None,
) -> requests.Response | None:
    ranked: list[tuple[int, str]] = []
    for href, text in parser.links:
        resolved = resolve_download_href(href)
        if resolved is None:
            continue
        joined = (resolved + " " + text).lower()
        score = 0
        if "download" in joined:
            score += 5
        if any(
            extension in joined
            for extension in _EXTENSIONS
        ):
            score += 4
        if "data" in joined:
            score += 1
        if score:
            ranked.append((score, resolved))

    for _, href in sorted(
        ranked,
        reverse=True,
    ):
        candidate = session.get(
            urljoin(response.url, href),
            timeout=(
                max(1.0, min(float(timeout), 30.0)),
                None,
            ),
            stream=True,
        )
        candidate.raise_for_status()
        if run_log is not None:
            run_log.event(
                "download_candidate",
                "respuesta",
                url=candidate.url,
                status=candidate.status_code,
                content_type=candidate.headers.get(
                    "content-type",
                    "",
                ),
                content_length=candidate.headers.get(
                    "content-length",
                    "",
                ),
            )
        clean_href = href.lower().split("?", 1)[0]
        extension_match = any(
            clean_href.endswith(extension)
            for extension in _EXTENSIONS
        )
        if (
            looks_downloadable(candidate)
            or extension_match
        ):
            return candidate
    return None
