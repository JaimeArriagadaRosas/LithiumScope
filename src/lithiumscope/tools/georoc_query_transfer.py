from __future__ import annotations

import threading
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

from lithiumscope.runtime.console_status import Spinner
from lithiumscope.tools.georoc_query_log import BoundedRunLog
from lithiumscope.tools.georoc_query_models import Form


def _format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB")
    size = float(max(0, value))
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return (
                f"{size:.0f} {unit}"
                if unit == "B"
                else f"{size:.1f} {unit}"
            )
        size /= 1024.0
    return f"{size:.1f} GB"


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def request_compiled_export(
    session: requests.Session,
    base_url: str,
    form: Form,
    payload: list[tuple[str, str]],
    connect_timeout: float,
    spinner: Spinner,
    run_log: BoundedRunLog,
) -> requests.Response:
    url = urljoin(
        base_url,
        form.action or base_url,
    )
    started = time.monotonic()
    stop = threading.Event()

    def update_elapsed() -> None:
        while not stop.wait(1.0):
            spinner.update(
                "GEOROC → compilando extracción "
                f"({_format_elapsed(time.monotonic() - started)} transcurridos)"
            )

    spinner.update(
        "GEOROC → compilando extracción (00:00 transcurridos)"
    )
    run_log.event(
        "compile_request",
        "inicio",
        method=form.method.upper(),
        url=url,
    )
    thread = threading.Thread(
        target=update_elapsed,
        name="georoc-compile-elapsed",
        daemon=True,
    )
    thread.start()
    try:
        request_timeout = (
            max(1.0, float(connect_timeout)),
            None,
        )
        if form.method == "post":
            response = session.post(
                url,
                data=payload,
                timeout=request_timeout,
                stream=False,
            )
        else:
            response = session.get(
                url,
                params=payload,
                timeout=request_timeout,
                stream=False,
            )
        response.raise_for_status()
        run_log.event(
            "compile_request",
            "respuesta completa recibida",
            status=response.status_code,
            url=response.url,
            content_type=response.headers.get(
                "content-type",
                "",
            ),
            content_length=response.headers.get(
                "content-length",
                "",
            ),
            elapsed=_format_elapsed(
                time.monotonic() - started
            ),
        )
        return response
    except BaseException as exc:
        run_log.event(
            "compile_request",
            "error",
            error_type=type(exc).__name__,
            detail=str(exc),
            url=url,
            elapsed=_format_elapsed(
                time.monotonic() - started
            ),
        )
        raise
    finally:
        stop.set()
        thread.join(timeout=2.0)


def stream_response_to_file(
    response: requests.Response,
    path: Path,
    spinner: Spinner,
    run_log: BoundedRunLog,
    *,
    chunk_size: int = 1024 * 1024,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    total_header = response.headers.get(
        "content-length",
        "",
    )
    try:
        total = int(total_header)
    except (TypeError, ValueError):
        total = 0

    downloaded = 0
    started = time.monotonic()
    last_update = 0.0
    run_log.event(
        "download",
        "inicio",
        url=response.url,
        total_bytes=total or None,
    )

    try:
        with path.open("wb") as handle:
            for chunk in response.iter_content(
                chunk_size=max(64 * 1024, chunk_size)
            ):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_update < 0.2:
                    continue
                last_update = now
                elapsed = max(0.001, now - started)
                rate = downloaded / elapsed

                if total > 0:
                    percent = min(
                        100.0,
                        downloaded * 100.0 / total,
                    )
                    spinner.update(
                        "GEOROC → descargando "
                        f"{_format_bytes(downloaded)} / "
                        f"{_format_bytes(total)} "
                        f"({percent:5.1f}%) | "
                        f"{_format_bytes(int(rate))}/s"
                    )
                else:
                    spinner.update(
                        "GEOROC → descargando "
                        f"{_format_bytes(downloaded)} | "
                        f"{_format_bytes(int(rate))}/s | "
                        f"{_format_elapsed(elapsed)}"
                    )

        run_log.event(
            "download",
            "completada",
            downloaded_bytes=downloaded,
            total_bytes=total or None,
            elapsed=_format_elapsed(
                time.monotonic() - started
            ),
        )
        return path
    except BaseException as exc:
        path.unlink(missing_ok=True)
        run_log.event(
            "download",
            "error",
            error_type=type(exc).__name__,
            detail=str(exc),
            downloaded_bytes=downloaded,
        )
        raise
