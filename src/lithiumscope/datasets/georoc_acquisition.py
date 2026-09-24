from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import md5
import json
from pathlib import Path
import re
import time
from typing import Iterable

import requests

from lithiumscope.core.logger import get_logger
from lithiumscope.runtime.console_status import Spinner

logger = get_logger("datasets.georoc_acquisition")

GEOROC_DATASET_PID = "doi:10.25625/PVFZCE"
GEOROC_DATASET_DOI = "https://doi.org/10.25625/PVFZCE"
GEOROC_METADATA_URL = (
    "https://georoc.eu/georoc/precompiled/"
    "metadata.php?doi=10.25625/PVFZCE"
)
DATAVERSE_BASE_URL = (
    "https://data.goettingen-research-online.de"
)

_ANDEAN_PATTERN = re.compile(
    r"_ANDEAN_ARC_part([123])\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GeorocRemoteFile:
    part: int
    file_id: int
    filename: str
    size_bytes: int
    persistent_id: str
    checksum_type: str | None = None
    checksum_value: str | None = None

    @property
    def download_url(self) -> str:
        return (
            f"{DATAVERSE_BASE_URL}/api/access/datafile/"
            f"{self.file_id}?format=original"
        )

    @property
    def size_gib(self) -> float:
        return self.size_bytes / (1024 ** 3)


@dataclass(frozen=True)
class GeorocDownloadResult:
    remote: GeorocRemoteFile
    local_path: Path
    resumed_from_bytes: int
    downloaded_bytes: int
    verified: bool


def _dataset_api_url() -> str:
    return (
        f"{DATAVERSE_BASE_URL}/api/datasets/"
        ":persistentId/"
    )


def _datafile_payload(entry: dict) -> dict:
    data_file = entry.get("dataFile")
    return data_file if isinstance(data_file, dict) else {}


def parse_andean_arc_files(
    payload: dict,
) -> list[GeorocRemoteFile]:
    data = payload.get("data", payload)
    version = data.get("latestVersion", {})
    entries = version.get("files", [])
    result: list[GeorocRemoteFile] = []

    for entry in entries:
        data_file = _datafile_payload(entry)
        filename = str(
            data_file.get("filename")
            or entry.get("label")
            or ""
        ).strip()
        match = _ANDEAN_PATTERN.search(filename)
        if match is None:
            continue

        file_id = data_file.get("id")
        size = data_file.get("filesize")
        persistent_id = str(
            data_file.get("persistentId") or ""
        ).strip()
        if file_id is None or size is None:
            continue

        checksum = data_file.get("checksum") or {}
        checksum_type = checksum.get("type")
        checksum_value = checksum.get("value")

        result.append(
            GeorocRemoteFile(
                part=int(match.group(1)),
                file_id=int(file_id),
                filename=filename,
                size_bytes=int(size),
                persistent_id=persistent_id,
                checksum_type=(
                    str(checksum_type)
                    if checksum_type
                    else None
                ),
                checksum_value=(
                    str(checksum_value)
                    if checksum_value
                    else None
                ),
            )
        )

    result.sort(key=lambda item: item.part)
    parts = [item.part for item in result]
    if parts != [1, 2, 3]:
        raise RuntimeError(
            "La API oficial de GEOROC no devolvió las tres "
            "partes esperadas de ANDEAN_ARC. "
            f"Partes encontradas: {parts or 'ninguna'}."
        )
    return result


def discover_andean_arc_files(
    *,
    session=requests,
    timeout_seconds: float = 60.0,
) -> list[GeorocRemoteFile]:
    response = session.get(
        _dataset_api_url(),
        params={
            "persistentId": GEOROC_DATASET_PID,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if (
        isinstance(payload, dict)
        and payload.get("status")
        and payload.get("status") != "OK"
    ):
        raise RuntimeError(
            "Dataverse devolvió un estado no exitoso: "
            f"{payload.get('status')}"
        )
    return parse_andean_arc_files(payload)


def _human_bytes(value: int) -> str:
    amount = float(max(0, value))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    return f"{amount:.2f} {unit}"


def _hash_file_md5(path: Path) -> str:
    digest = md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(
    remote: GeorocRemoteFile,
    path: Path,
) -> bool:
    checksum_type = (
        remote.checksum_type or ""
    ).strip().upper()
    checksum_value = (
        remote.checksum_value or ""
    ).strip().lower()

    if not checksum_value:
        return False
    if checksum_type != "MD5":
        logger.warning(
            "Checksum type %s is not supported for %s",
            checksum_type,
            remote.filename,
        )
        return False

    actual = _hash_file_md5(path)
    if actual.lower() != checksum_value:
        raise RuntimeError(
            "Checksum MD5 incorrecto para "
            f"{remote.filename}: esperado={checksum_value}, "
            f"obtenido={actual}."
        )
    return True


def download_georoc_file(
    remote: GeorocRemoteFile,
    destination_dir: Path,
    *,
    session=requests,
    verify_checksum: bool = False,
    force: bool = False,
    chunk_size: int = 8 * 1024 * 1024,
    connect_timeout_seconds: float = 30.0,
    read_timeout_seconds: float = 180.0,
) -> GeorocDownloadResult:
    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination = destination_dir / remote.filename
    partial = destination.with_suffix(
        destination.suffix + ".part"
    )

    if destination.is_file() and not force:
        local_size = destination.stat().st_size
        if local_size == remote.size_bytes:
            verified = (
                _verify_checksum(remote, destination)
                if verify_checksum
                else False
            )
            return GeorocDownloadResult(
                remote=remote,
                local_path=destination,
                resumed_from_bytes=local_size,
                downloaded_bytes=0,
                verified=verified,
            )
        raise RuntimeError(
            f"{destination.name} ya existe pero su tamaño "
            f"({local_size}) no coincide con el oficial "
            f"({remote.size_bytes}). Use --force para reemplazarlo."
        )

    if force:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    resume_from = (
        partial.stat().st_size
        if partial.is_file()
        else 0
    )
    if resume_from > remote.size_bytes:
        raise RuntimeError(
            f"El archivo parcial {partial.name} es mayor "
            "que el archivo remoto. Use --force."
        )
    if resume_from == remote.size_bytes and resume_from > 0:
        partial.replace(destination)
        verified = (
            _verify_checksum(remote, destination)
            if verify_checksum
            else False
        )
        return GeorocDownloadResult(
            remote=remote,
            local_path=destination,
            resumed_from_bytes=resume_from,
            downloaded_bytes=0,
            verified=verified,
        )

    headers = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    spinner = Spinner(
        f"GEOROC parte {remote.part}: "
        f"{_human_bytes(resume_from)}/"
        f"{_human_bytes(remote.size_bytes)}"
    ).start()

    downloaded_this_run = 0
    mode = "ab" if resume_from > 0 else "wb"
    request_start = resume_from
    last_update = 0.0

    try:
        response = session.get(
            remote.download_url,
            headers=headers,
            stream=True,
            timeout=(
                connect_timeout_seconds,
                read_timeout_seconds,
            ),
        )

        if (
            resume_from > 0
            and response.status_code == 200
        ):
            logger.warning(
                "Server ignored Range for %s; restarting download.",
                remote.filename,
            )
            mode = "wb"
            request_start = 0
            resume_from = 0
        elif (
            resume_from > 0
            and response.status_code != 206
        ):
            response.raise_for_status()
        else:
            response.raise_for_status()

        with partial.open(mode) as handle:
            for chunk in response.iter_content(
                chunk_size=max(
                    1024 * 1024,
                    int(chunk_size),
                )
            ):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded_this_run += len(chunk)

                now = time.monotonic()
                if now - last_update >= 0.5:
                    current = (
                        request_start
                        + downloaded_this_run
                    )
                    percent = (
                        100.0
                        * current
                        / remote.size_bytes
                        if remote.size_bytes
                        else 0.0
                    )
                    spinner.update(
                        f"GEOROC parte {remote.part}: "
                        f"{_human_bytes(current)}/"
                        f"{_human_bytes(remote.size_bytes)} "
                        f"({percent:.1f}%)"
                    )
                    last_update = now

        final_size = partial.stat().st_size
        if final_size != remote.size_bytes:
            raise RuntimeError(
                f"Descarga incompleta de {remote.filename}: "
                f"{final_size} bytes de {remote.size_bytes}."
            )

        partial.replace(destination)
        verified = (
            _verify_checksum(remote, destination)
            if verify_checksum
            else False
        )
        spinner.succeed(
            f"GEOROC parte {remote.part} lista: "
            f"{_human_bytes(remote.size_bytes)}"
        )
        return GeorocDownloadResult(
            remote=remote,
            local_path=destination,
            resumed_from_bytes=request_start,
            downloaded_bytes=downloaded_this_run,
            verified=verified,
        )
    except Exception:
        spinner.fail(
            f"GEOROC parte {remote.part} interrumpida; "
            "el .part queda disponible para reanudar."
        )
        raise


def download_andean_arc(
    destination_dir: Path,
    *,
    parts: Iterable[int] = (1, 2, 3),
    verify_checksum: bool = False,
    force: bool = False,
    session=requests,
) -> list[GeorocDownloadResult]:
    requested = {
        int(value)
        for value in parts
    }
    remote_files = discover_andean_arc_files(
        session=session
    )
    selected = [
        item
        for item in remote_files
        if item.part in requested
    ]
    if not selected:
        raise ValueError(
            "No se seleccionaron partes GEOROC válidas."
        )

    results: list[GeorocDownloadResult] = []
    for remote in selected:
        results.append(
            download_georoc_file(
                remote,
                destination_dir,
                session=session,
                verify_checksum=verify_checksum,
                force=force,
            )
        )

    manifest = {
        "dataset_pid": GEOROC_DATASET_PID,
        "dataset_doi": GEOROC_DATASET_DOI,
        "metadata_url": GEOROC_METADATA_URL,
        "destination": str(destination_dir),
        "files": [
            {
                **asdict(result.remote),
                "local_path": str(result.local_path),
                "resumed_from_bytes": (
                    result.resumed_from_bytes
                ),
                "downloaded_bytes": (
                    result.downloaded_bytes
                ),
                "verified": result.verified,
            }
            for result in results
        ],
    }
    manifest_path = (
        destination_dir
        / "georoc_andean_arc_download.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return results
