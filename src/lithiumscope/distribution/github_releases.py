from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Callable

import requests

from lithiumscope.core.config import load_config
from lithiumscope.core.logger import get_logger

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
        if release.get("draft") or not tag_name.startswith(tag_prefix):
            continue

        assets = [
            asset
            for asset in release.get("assets", [])
            if str(asset.get("name", "")).endswith(asset_suffix)
            and asset.get("browser_download_url")
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

    response = requests.get(
        f"{api_base}/repos/{repository}/releases",
        params={"per_page": min(100, max(1, limit))},
        headers=_headers(),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(
            "GitHub Releases devolvió una respuesta inesperada."
        )
    return parse_model_releases(
        payload,
        asset_suffix=asset_suffix,
        tag_prefix=tag_prefix,
    )[:limit]


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
    if digest and digest.startswith("sha256:"):
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
    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    try:
        with requests.get(
            release.asset_url,
            headers=_headers(),
            stream=True,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            total = int(
                response.headers.get("content-length")
                or release.asset_size
                or 0
            )
            if total > max_bytes:
                raise RuntimeError(
                    "El asset reportado excede el tamaño máximo permitido."
                )

            downloaded = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise RuntimeError(
                            "La descarga excedió el tamaño máximo permitido."
                        )
                    handle.write(chunk)
                    if progress_callback is not None:
                        progress_callback(
                            downloaded,
                            total,
                        )
        temporary.replace(destination)
        digest = _verify_github_digest(
            destination,
            release.asset_digest,
        )
        logger.info(
            "Downloaded model release tag=%s asset=%s sha256=%s",
            release.tag_name,
            release.asset_name,
            digest,
        )
        return destination, digest
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
