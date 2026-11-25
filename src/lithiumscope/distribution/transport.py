"""Bounded HTTPS recovery; certificate verification is never disabled."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlsplit

import requests

from lithiumscope.core.logging.session import get_logger

logger = get_logger("distribution.transport")


def retryable(exc):
    if isinstance(exc, requests.exceptions.HTTPError):
        return exc.response is not None and exc.response.status_code in {408,429,500,502,503,504}
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                            requests.exceptions.ChunkedEncodingError))


def retry(action, *, attempts=3):
    for i in range(attempts):
        try:
            return action()
        except requests.RequestException as exc:
            if not retryable(exc) or i == attempts-1:
                raise
            logger.warning("HTTPS attempt %s/%s failed: %s", i+1,attempts,type(exc).__name__)
            time.sleep(min(2**i,4))


def curl_public(url, destination, *, timeout, max_bytes):
    """Alternative native HTTPS stack (e.g. Windows Schannel), public URLs only."""
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if not executable:
        raise RuntimeError("curl no está disponible en PATH")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in {"github.com","api.github.com"} or parsed.username or parsed.password:
        raise ValueError("Fallback curl solo admite URLs HTTPS públicas de GitHub")
    command = [executable,"--disable","--silent","--show-error","--fail","--location",
               "--proto","=https","--proto-redir","=https","--http1.1",
               "--connect-timeout",str(min(timeout,15)),"--max-time",str(timeout),
               "--max-filesize",str(max_bytes),"--user-agent","LithiumScope",
               "--output",str(destination),url]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout+5, check=False)
    if result.returncode:
        raise RuntimeError(f"curl falló (código {result.returncode}); revise red, proxy y certificados")
    if not Path(destination).is_file() or Path(destination).stat().st_size > max_bytes:
        raise RuntimeError("Respuesta curl ausente o demasiado grande")


def catalog_json(url, headers, timeout):
    def fetch():
        with requests.get(url,headers=headers,timeout=(min(timeout,15),timeout)) as response:
            response.raise_for_status()
            return response.json()
    try:
        return retry(fetch)
    except requests.RequestException as exc:
        if not retryable(exc):
            raise
        logger.warning("API requests no disponible (%s); probando curl HTTPS público",type(exc).__name__)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"catalog.json"
            try:
                curl_public(url,path,timeout=timeout,max_bytes=10*1024*1024)
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as fallback:
                raise RuntimeError(f"GitHub API no disponible: {type(exc).__name__}; alternativa curl: {fallback}. "
                                   "No se deshabilitó TLS.") from exc


def explain_error(exc):
    if isinstance(exc,requests.exceptions.SSLError) or "SSL" in str(exc) or "TLS" in str(exc):
        return "Falló la conexión TLS. Puede ser red, proxy, antivirus o certificados; no demuestra un cambio de GitHub."
    if isinstance(exc,requests.exceptions.HTTPError) and exc.response is not None:
        return f"GitHub respondió HTTP {exc.response.status_code}; revise permisos, límite de API o disponibilidad."
    return f"{type(exc).__name__}: no se pudo completar la consulta/transferencia."
