from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import shutil
import subprocess

from lithiumscope.core.config import load_config
from lithiumscope.core.device import detect_device
from lithiumscope.core.paths import DATA_DIR, PROJECT_ROOT
from lithiumscope.datasets.downloader import ensure_dataset
from lithiumscope.datasets.georoc_filtered_acquisition import (
    acquire_filtered_georoc,
    write_acquisition_contract,
)
from lithiumscope.runtime.preboot import run_preboot


@dataclass(frozen=True)
class LabPrebootReport:
    base_dataset: Path
    georoc_files: tuple[Path, ...]


@dataclass(frozen=True)
class GpuProbeResult:
    nvidia_smi: bool
    devices: tuple[str, ...]
    memory: tuple[str, ...]
    torch_cuda: bool
    xgboost_cuda_test: bool | None
    catboost_gpu_test: bool | None
    notes: tuple[str, ...]


def _project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _georoc_files() -> tuple[Path, ...]:
    cfg = load_config("model_1")
    georoc = cfg.get("data_sources", {}).get("georoc", {})
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
            for path in pattern.parent.glob(pattern.name)
            if path.is_file() and path.stat().st_size > 0
        )
    )


def run_lab_preboot(
    *,
    acquire_data: bool = True,
    verbose: bool = True,
) -> LabPrebootReport:
    report = run_preboot(
        verbose=verbose,
        provision_datasets=False,
    )
    if not report.core_ready:
        raise RuntimeError(
            "El preboot base no está listo. Revise el reporte indicado."
        )

    base = ensure_dataset("mamani09_public_mirror")
    georoc_files = _georoc_files()

    if not georoc_files and acquire_data:
        print(
            "\n[LAB] GEOROC filtrado no está disponible; "
            "se intentará una consulta remota limitada a los datos "
            "que usa LithiumScope."
        )
        target_dir = (
            DATA_DIR
            / "raw"
            / "model_1"
            / "georoc"
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
        georoc_files = _georoc_files()

    if not georoc_files:
        raise RuntimeError(
            "El laboratorio requiere una extracción GEOROC filtrada "
            "y no hay CSV disponible. No se descargará el precompilado "
            "Andean Arc completo como fallback."
        )

    if verbose:
        print(
            "\n=== PREBOOT LABORATORIO ==="
        )
        print(f"  Mamani09 bruto      [OK] {base}")
        print(
            "  GEOROC bruto        [OK] "
            f"{len(georoc_files)} archivo(s)"
        )
        for path in georoc_files:
            print(
                f"    - {path.name} "
                f"({path.stat().st_size / 1048576:.1f} MiB)"
            )
        print("=" * 29)

    return LabPrebootReport(
        base_dataset=base,
        georoc_files=georoc_files,
    )


def _nvidia_smi_probe() -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False, (), ()
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return False, (), ()
    if completed.returncode != 0:
        return False, (), ()

    devices: list[str] = []
    memory: list[str] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        pieces = [piece.strip() for piece in line.split(",", 1)]
        devices.append(pieces[0])
        memory.append(
            f"{pieces[1]} MiB"
            if len(pieces) > 1
            else "desconocida"
        )
    return bool(devices), tuple(devices), tuple(memory)


def _test_xgboost_cuda() -> tuple[bool | None, str | None]:
    if importlib.util.find_spec("xgboost") is None:
        return None, "XGBoost no está instalado."
    try:
        import numpy as np
        import xgboost as xgb

        x = np.array(
            [[0.0], [1.0], [2.0], [3.0]],
            dtype=float,
        )
        y = np.array(
            [0.0, 1.0, 2.0, 3.0],
            dtype=float,
        )
        model = xgb.XGBRegressor(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        model.fit(x, y)
        return True, None
    except Exception as exc:
        return False, f"XGBoost CUDA falló: {exc}"


def _test_catboost_gpu() -> tuple[bool | None, str | None]:
    if importlib.util.find_spec("catboost") is None:
        return None, "CatBoost no está instalado."
    try:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=1,
            depth=2,
            task_type="GPU",
            devices="0",
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            [[0.0], [1.0], [2.0], [3.0]],
            [0.0, 1.0, 2.0, 3.0],
        )
        return True, None
    except Exception as exc:
        return False, f"CatBoost GPU falló: {exc}"


def probe_gpu() -> GpuProbeResult:
    # Reuse the project's own detector, then prove actual backend execution.
    info = detect_device(prefer_gpu=True)
    smi, devices, memory = _nvidia_smi_probe()
    notes: list[str] = []

    torch_cuda = bool(info.cuda_available)
    xgb_ok, xgb_note = _test_xgboost_cuda()
    cat_ok, cat_note = _test_catboost_gpu()
    if xgb_note:
        notes.append(xgb_note)
    if cat_note:
        notes.append(cat_note)
    if info.diagnostic:
        notes.append(info.diagnostic)

    return GpuProbeResult(
        nvidia_smi=smi,
        devices=devices,
        memory=memory,
        torch_cuda=torch_cuda,
        xgboost_cuda_test=xgb_ok,
        catboost_gpu_test=cat_ok,
        notes=tuple(notes),
    )


def print_gpu_probe() -> int:
    result = probe_gpu()

    def state(value: bool | None) -> str:
        if value is None:
            return "NO INSTALADO"
        return "OK" if value else "NO"

    print("\n=== DIAGNÓSTICO GPU LITHIUMSCOPE ===")
    print(
        f"  NVIDIA / nvidia-smi  "
        f"[{'OK' if result.nvidia_smi else 'NO'}]"
    )
    if result.devices:
        for index, name in enumerate(result.devices):
            mem = (
                result.memory[index]
                if index < len(result.memory)
                else "desconocida"
            )
            print(f"    GPU {index}: {name} | VRAM {mem}")
    print(
        f"  PyTorch CUDA         "
        f"[{'OK' if result.torch_cuda else 'NO'}]"
    )
    print(
        f"  XGBoost prueba CUDA  "
        f"[{state(result.xgboost_cuda_test)}]"
    )
    print(
        f"  CatBoost prueba GPU  "
        f"[{state(result.catboost_gpu_test)}]"
    )
    for note in result.notes:
        print(f"  Nota                 {note}")
    print("=" * 36)

    usable = (
        result.torch_cuda
        or result.xgboost_cuda_test is True
        or result.catboost_gpu_test is True
    )
    return 0 if usable else 1
