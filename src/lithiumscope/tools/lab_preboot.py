from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import DATA_DIR, PROJECT_ROOT
from lithiumscope.datasets.downloader import ensure_dataset
from lithiumscope.datasets.georoc_filtered_acquisition import (
    GEOROC_FILTERED_NAME,
    acquire_filtered_georoc,
    write_acquisition_contract,
)
from lithiumscope.datasets.registry import get_dataset_spec
from lithiumscope.runtime.preboot import run_preboot
from lithiumscope.tools.georoc_source_checkpoint import (
    source_path_for,
)


@dataclass(frozen=True)
class LabPrebootReport:
    base_dataset: Path
    georoc_files: tuple[Path, ...]


def _project_path(raw: str) -> Path:
    path = Path(raw)
    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )


def _georoc_files() -> tuple[Path, ...]:
    cfg = load_config("model_1")
    georoc = cfg.get(
        "data_sources",
        {},
    ).get("georoc", {})
    raw_glob = str(
        georoc.get(
            "input_glob",
            "data/raw/model_1/georoc/*.csv",
        )
    )
    pattern = _project_path(raw_glob)
    return tuple(
        sorted(
            path
            for path in pattern.parent.glob(
                pattern.name
            )
            if (
                path.is_file()
                and path.stat().st_size > 0
            )
        )
    )


def _ensure_mamani(verbose: bool) -> Path:
    spec = get_dataset_spec(
        "mamani09_public_mirror"
    )
    candidate = (
        DATA_DIR
        / "raw"
        / spec.model
        / spec.destination_name
    )
    found = (
        candidate.is_file()
        and candidate.stat().st_size > 0
    )
    if verbose:
        print(
            "[LAB] Mamani09 "
            + (
                "[ENCONTRADO]"
                if found
                else "[NO ENCONTRADO] → descargando"
            )
        )
    return ensure_dataset(
        "mamani09_public_mirror"
    )


def _ensure_georoc(
    *,
    acquire_data: bool,
    verbose: bool,
) -> tuple[Path, ...]:
    files = _georoc_files()
    if files:
        if verbose:
            print(
                "[LAB] GEOROC [ENCONTRADO] "
                f"{len(files)} archivo(s)"
            )
        return files

    if not acquire_data:
        raise RuntimeError(
            "GEOROC [NO ENCONTRADO]."
        )

    target_dir = (
        DATA_DIR
        / "raw"
        / "model_1"
        / "georoc"
    )
    destination = (
        target_dir
        / GEOROC_FILTERED_NAME
    )
    checkpoint = source_path_for(
        destination
    )

    if verbose:
        if (
            checkpoint.is_file()
            and checkpoint.stat().st_size > 0
        ):
            print(
                "[LAB] GEOROC [FUENTE DESCARGADA] "
                "→ retomando procesamiento local"
            )
        else:
            print(
                "[LAB] GEOROC [NO ENCONTRADO] "
                "→ descargando extracción filtrada"
            )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    write_acquisition_contract(
        target_dir
        / "lithiumscope_georoc_query_contract.json"
    )
    acquire_filtered_georoc()
    files = _georoc_files()
    if not files:
        raise RuntimeError(
            "GEOROC no fue descargado."
        )
    return files


def run_lab_preboot(
    *,
    acquire_data: bool = True,
    verbose: bool = True,
) -> LabPrebootReport:
    report = run_preboot(
        verbose=False,
        provision_datasets=False,
    )
    if not report.core_ready:
        raise RuntimeError(
            "El entorno base no supera el preboot."
        )

    if verbose:
        print("\n=== PREBOOT LABORATORIO ===")
        print("[LAB] Entorno [OK]")

    base = _ensure_mamani(verbose)
    georoc_files = _ensure_georoc(
        acquire_data=acquire_data,
        verbose=verbose,
    )

    if verbose:
        print("[LAB] Datos brutos [OK]")
        print("=" * 29)

    return LabPrebootReport(
        base_dataset=base,
        georoc_files=georoc_files,
    )
