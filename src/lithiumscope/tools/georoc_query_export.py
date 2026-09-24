from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin
import zipfile

import pandas as pd
import requests

from lithiumscope.tools.georoc_query_html import (
    FormParser,
    norm,
)


def looks_downloadable(
    response: requests.Response,
) -> bool:
    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()
    disposition = response.headers.get(
        "content-disposition",
        "",
    ).lower()
    return (
        "text/csv" in content_type
        or "application/csv" in content_type
        or "spreadsheet" in content_type
        or "excel" in content_type
        or "application/zip" in content_type
        or "attachment" in disposition
    )


def materialize_download(
    response: requests.Response,
    destination: Path,
) -> Path:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()
    disposition = response.headers.get(
        "content-disposition",
        "",
    ).lower()

    if (
        "zip" in content_type
        or ".zip" in disposition
    ):
        with zipfile.ZipFile(
            BytesIO(response.content)
        ) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith(
                    (
                        ".csv",
                        ".txt",
                        ".xlsx",
                        ".xls",
                    )
                )
            ]
            if not candidates:
                raise RuntimeError(
                    "La exportación GEOROC ZIP no contiene "
                    "una tabla compatible."
                )
            raw = archive.read(candidates[0])
            suffix = Path(candidates[0]).suffix.lower()
    else:
        raw = response.content
        suffix = (
            ".xlsx"
            if (
                "excel" in content_type
                or ".xlsx" in disposition
            )
            else ".csv"
        )

    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(BytesIO(raw))
    else:
        temporary = destination.with_suffix(
            ".download"
        )
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


def validate_export(
    path: Path,
) -> None:
    try:
        frame = pd.read_csv(path, nrows=20)
    except Exception as exc:
        raise RuntimeError(
            "La exportación GEOROC no pudo leerse "
            f"como CSV: {exc}"
        ) from exc

    normalized = {
        norm(column)
        for column in frame.columns
    }
    required_groups = {
        "LI": ("LI", "LIPPM"),
        "LONGITUDE": (
            "LONGITUDE",
            "LONGITUDEMIN",
        ),
        "LATITUDE": (
            "LATITUDE",
            "LATITUDEMIN",
        ),
    }
    missing = [
        label
        for label, aliases in required_groups.items()
        if not any(
            any(alias in column for alias in aliases)
            for column in normalized
        )
    ]
    if missing:
        raise RuntimeError(
            "La exportación obtenida no corresponde "
            "al contrato LithiumScope. Faltan: "
            + ", ".join(missing)
        )


def find_download_link(
    session: requests.Session,
    response: requests.Response,
    parser: FormParser,
    timeout: float,
) -> requests.Response | None:
    ranked: list[tuple[int, str]] = []
    for href, text in parser.links:
        joined = (
            href + " " + text
        ).lower()
        score = 0
        if "download" in joined:
            score += 5
        if any(
            extension in joined
            for extension in (
                ".csv",
                ".txt",
                ".xlsx",
                ".xls",
                ".zip",
            )
        ):
            score += 4
        if "data" in joined:
            score += 1
        if score:
            ranked.append((score, href))

    for _, href in sorted(
        ranked,
        reverse=True,
    ):
        candidate = session.get(
            urljoin(response.url, href),
            timeout=timeout,
        )
        candidate.raise_for_status()
        if looks_downloadable(candidate):
            return candidate
    return None
