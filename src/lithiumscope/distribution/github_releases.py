from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from dataclasses import asdict
from urllib.parse import urlencode, urlsplit
from uuid import uuid4
import os
from pathlib import Path
from typing import Callable

import requests

from lithiumscope.core.config import load_config
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.paths import MODELS_DIR
from lithiumscope.distribution.transport import catalog_json, curl_public, retry, retryable

logger = get_logger("distribution.github_releases")


@dataclass(frozen=True)
class ModelRelease:
    tag_name: str
    release_name: str
    published_at: str | None
    html_url: str
    asset_name: str
    asset_url: str
    asset_size: int
    asset_digest: str | None


def _github_config() -> dict:
    return load_config("distribution")["github"]


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "LithiumScope",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_model_releases(
    payload: list[dict],
    *,
    asset_suffix: str,
    tag_prefix: str,
) -> list[ModelRelease]:
    releases: list[ModelRelease] = []
    for release in payload:
        tag_name = str(release.get("tag_name", ""))
        if release.get("draft") or release.get("prerelease") or not tag_name.startswith(tag_prefix):
            continue

        assets = [
            asset
            for asset in release.get("assets", [])
            if str(asset.get("name", "")).endswith(asset_suffix)
            and asset.get("browser_download_url")
            and Path(str(asset.get("name",""))).name == asset.get("name")
            and not any(c in str(asset.get("name","")) for c in ("/","\\",":",".."))
        ]
        if not assets:
            continue

        asset = assets[0]
        releases.append(
            ModelRelease(
                tag_name=tag_name,
                release_name=str(release.get("name") or tag_name),
                published_at=release.get("published_at"),
                html_url=str(release.get("html_url", "")),
                asset_name=str(asset["name"]),
                asset_url=str(asset["browser_download_url"]),
                asset_size=int(asset.get("size") or 0),
                asset_digest=(
                    str(asset["digest"])
                    if asset.get("digest")
                    else None
                ),
            )
        )
    return releases


def list_model_releases() -> list[ModelRelease]:
    cfg = _github_config()
    repository = str(cfg["repository"])
    api_base = str(
        cfg.get("api_base", "https://api.github.com")
    ).rstrip("/")
    limit = int(cfg.get("max_releases", 20))
    timeout = int(cfg.get("request_timeout_seconds", 60))
    asset_suffix = str(
        cfg.get("asset_suffix", "-artifacts.zip")
    )
    tag_prefix = str(
        cfg.get("tag_prefix", "lithiumscope-training_")
    )

    url = f"{api_base}/repos/{repository}/releases?" + urlencode({"per_page": min(100,max(1,limit))})
    payload = catalog_json(url,_headers(),timeout)
    if not isinstance(payload, list):
        raise RuntimeError(
            "GitHub Releases devolvió una respuesta inesperada."
        )
    releases = sorted(parse_model_releases(payload,asset_suffix=asset_suffix,tag_prefix=tag_prefix),
                      key=lambda item:item.published_at or "",reverse=True)[:limit]
    try:
        cache = MODELS_DIR/"release_catalog.json"
        cache.parent.mkdir(parents=True,exist_ok=True)
        from datetime import datetime, timezone
        tmp=cache.with_name(f".catalog-{uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps({"repository":repository,"checked_at":datetime.now(timezone.utc).isoformat(),
                "releases":[asdict(r) for r in releases]},ensure_ascii=False,indent=2),encoding="utf-8")
            tmp.replace(cache)
        finally:
            tmp.unlink(missing_ok=True)
    except OSError:
        logger.warning("No fue posible guardar el catálogo local")
    return releases


def cached_model_releases():
    path = MODELS_DIR/"release_catalog.json"
    if not path.is_file():
        return [], None
    data=json.loads(path.read_text(encoding="utf-8"))
    if data["repository"] != _github_config()["repository"]:
        raise ValueError("Catálogo local de otro repositorio")
    return [ModelRelease(**r) for r in data["releases"]],data["checked_at"]


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_github_digest(
    path: Path,
    digest: str | None,
) -> str:
    actual = _sha256(path)
    if digest:
        if not digest.startswith("sha256:") or len(digest.split(":",1)[1]) != 64:
            raise RuntimeError("Formato SHA-256 de GitHub inválido")
        expected = digest.split(":", 1)[1].lower()
        if actual.lower() != expected:
            raise RuntimeError(
                "El SHA-256 del asset descargado no coincide con GitHub. "
                f"esperado={expected} actual={actual}"
            )
    return actual


def download_release_asset(
    release: ModelRelease,
    destination: Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[Path, str]:
    cfg = _github_config()
    timeout = int(cfg.get("request_timeout_seconds", 60))
    max_bytes = int(
        float(cfg.get("max_asset_size_mb", 2048))
        * 1024
        * 1024
    )
    if release.asset_size > max_bytes:
        raise RuntimeError(
            "El bundle publicado excede el tamaño máximo permitido."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")

    if Path(release.asset_name).name!=release.asset_name or any(c in release.asset_name for c in ("/","\\",":","..")):
        raise ValueError("Nombre de asset inseguro")
    parsed=urlsplit(release.asset_url)
    if parsed.scheme!="https" or parsed.hostname!="github.com" or parsed.username or parsed.password:
        raise ValueError("URL de descarga no pertenece a GitHub HTTPS")
    expected_path=f"/{cfg['repository']}/releases/download/"
    if not parsed.path.startswith(expected_path):
        raise ValueError("El asset no pertenece al repositorio configurado")
    def transfer():
        with requests.get(release.asset_url,headers={"User-Agent":"LithiumScope"},
                          stream=True,timeout=(min(timeout,15),timeout)) as response:
            response.raise_for_status()
            total=int(response.headers.get("content-length") or release.asset_size or 0)
            if total>max_bytes:
                raise RuntimeError("Asset excede tamaño máximo")
            downloaded=0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if not chunk:
                        continue
                    downloaded+=len(chunk)
                    if downloaded>max_bytes:
                        raise RuntimeError("Descarga excede tamaño máximo")
                    handle.write(chunk)
                    if progress_callback:
                        progress_callback(downloaded,total)
            if total and downloaded!=total:
                raise requests.exceptions.ChunkedEncodingError("Descarga incompleta")
    try:
        try:
            retry(transfer)
        except requests.RequestException as exc:
            if not retryable(exc):
                raise
            logger.warning("Descarga requests falló; probando curl HTTPS público")
            temporary.unlink(missing_ok=True)
            curl_public(release.asset_url,temporary,timeout=max(timeout,300),max_bytes=max_bytes)
        if release.asset_size and temporary.stat().st_size!=release.asset_size:
            raise RuntimeError("Tamaño descargado no coincide con catálogo GitHub")
        digest=_verify_github_digest(temporary,release.asset_digest)
        import zipfile
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError("El asset no es un ZIP válido")
        temporary.replace(destination)
        logger.info("Downloaded model release tag=%s sha256=%s",release.tag_name,digest)
        return destination,digest
    finally:
        temporary.unlink(missing_ok=True)
