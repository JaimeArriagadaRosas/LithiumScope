from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin
import zipfile

import pandas as pd
import requests

from lithiumscope.runtime.console_status import Spinner
from lithiumscope.tools.georoc_query_download_links import (
    resolve_download_href,
)
from lithiumscope.tools.georoc_query_html import (
    FormParser,
    norm,
)
from lithiumscope.tools.georoc_query_log import BoundedRunLog
from lithiumscope.tools.georoc_query_transfer import (
    stream_response_to_file,
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
    *,
    spinner: Spinner | None = None,
    run_log: BoundedRunLog | None = None,
) -> Path:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    raw_path = destination.with_name(
        destination.name + ".download.part"
    )
    normalized_path = destination.with_name(
        destination.name + ".normalized.part"
    )
    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()
    disposition = response.headers.get(
        "content-disposition",
        "",
    ).lower()

    raw_path.unlink(missing_ok=True)
    normalized_path.unlink(missing_ok=True)

    try:
        if spinner is not None and run_log is not None:
            stream_response_to_file(
                response,
                raw_path,
                spinner,
                run_log,
            )
        else:
            raw_path.write_bytes(
                response.content
            )

        if run_log is not None:
            run_log.event(
                "normalize",
                "inicio",
                raw_bytes=raw_path.stat().st_size,
                content_type=content_type,
                disposition=disposition,
            )

        if (
            "zip" in content_type
            or ".zip" in disposition
        ):
            with zipfile.ZipFile(raw_path) as archive:
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
                member = candidates[0]
                raw = archive.read(member)
                suffix = Path(member).suffix.lower()
                if suffix in {".xlsx", ".xls"}:
                    frame = pd.read_excel(
                        BytesIO(raw)
                    )
                else:
                    extracted = raw_path.with_name(
                        raw_path.name + ".table"
                    )
                    extracted.write_bytes(raw)
                    try:
                        frame = pd.read_csv(
                            extracted,
                            sep=None,
                            engine="python",
                        )
                    finally:
                        extracted.unlink(
                            missing_ok=True
                        )
        elif (
            "excel" in content_type
            or ".xlsx" in disposition
            or ".xls" in disposition
        ):
            frame = pd.read_excel(raw_path)
        else:
            frame = pd.read_csv(
                raw_path,
                sep=None,
                engine="python",
            )

        frame.to_csv(
            normalized_path,
            index=False,
        )
        normalized_path.replace(destination)

        if run_log is not None:
            run_log.event(
                "normalize",
                "completada",
                rows=len(frame),
                columns=len(frame.columns),
                destination=destination,
            )
        return destination
    except BaseException as exc:
        normalized_path.unlink(
            missing_ok=True
        )
        if run_log is not None:
            run_log.event(
                "normalize",
                "error",
                error_type=type(exc).__name__,
                detail=str(exc),
            )
        raise
    finally:
        raw_path.unlink(missing_ok=True)


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
        resolved = resolve_download_href(
            href
        )
        if resolved is None:
            continue
        joined = (
            resolved + " " + text
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
        if looks_downloadable(candidate):
            return candidate
    return None
