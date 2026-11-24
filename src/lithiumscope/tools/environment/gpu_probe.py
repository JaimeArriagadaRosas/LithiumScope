from __future__ import annotations

from dataclasses import dataclass
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.resources.detection.host import detect_gpus
from lithiumscope.runtime.preboot import run_preboot


@dataclass(frozen=True)
class GpuProbeResult:
    nvidia_smi: bool
    devices: tuple[str, ...]
    memory: tuple[str, ...]
    torch_cuda: bool
    xgboost_cuda_test: bool | None
    catboost_gpu_test: bool | None
    notes: tuple[str, ...]


def probe_gpu() -> GpuProbeResult:
    info = detect_device(prefer_gpu=True)
    gpus = detect_gpus()
    return GpuProbeResult(
        bool(gpus), tuple(str(g["name"]) for g in gpus),
        tuple(str(g["vram_total_bytes"] // 1048576) + " MiB" if g["vram_total_bytes"] is not None else "N/D" for g in gpus),
        info.cuda_available, info.xgboost_cuda_available if gpus else None,
        info.catboost_gpu_available if gpus else None, (info.diagnostic,),
    )


def print_gpu_probe() -> int:
    preboot = run_preboot(
        verbose=False,
        provision_datasets=False,
    )
    if not preboot.core_ready:
        print(
            "[ERROR] El entorno base no supera "
            "el preboot."
        )
        return 2

    result = probe_gpu()

    def state(value: bool | None) -> str:
        if value is None:
            return "NO INSTALADO"
        return "OK" if value else "NO"

    print("\n=== DIAGNÓSTICO GPU LITHIUMSCOPE ===")
    print(
        "  NVIDIA / nvidia-smi  "
        f"[{'OK' if result.nvidia_smi else 'NO'}]"
    )
    for index, name in enumerate(result.devices):
        memory = (
            result.memory[index]
            if index < len(result.memory)
            else "desconocida"
        )
        print(
            f"    GPU {index}: {name} | "
            f"VRAM {memory}"
        )
    print(
        "  PyTorch CUDA         "
        f"[{'OK' if result.torch_cuda else 'NO'}]"
    )
    print(
        "  XGBoost prueba CUDA  "
        f"[{state(result.xgboost_cuda_test)}]"
    )
    print(
        "  CatBoost prueba GPU  "
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
