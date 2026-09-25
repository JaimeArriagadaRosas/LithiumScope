from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import platform
import shutil
import subprocess

from lithiumscope.core.logger import get_logger

logger = get_logger("device")


@dataclass(frozen=True)
class DeviceInfo:
    accelerator: str
    name: str
    cpu_count: int
    torch_available: bool
    cuda_available: bool
    mps_available: bool
    nvidia_driver_available: bool = False
    nvidia_device_names: tuple[str, ...] = ()
    xgboost_cuda_available: bool = False
    diagnostic: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.accelerator in {"cuda", "mps"}

    @property
    def any_cuda_backend(self) -> bool:
        return (
            self.cuda_available
            or self.xgboost_cuda_available
        )


def _detect_nvidia_driver() -> tuple[bool, tuple[str, ...]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False, ()

    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.SubprocessError):
        return False, ()

    if completed.returncode != 0:
        return False, ()

    names = tuple(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    )
    return bool(names), names


def _detect_xgboost_cuda() -> bool:
    if importlib.util.find_spec("xgboost") is None:
        return False
    try:
        import xgboost  # type: ignore

        build_info = getattr(
            xgboost,
            "build_info",
            lambda: {},
        )()
        value = build_info.get("USE_CUDA")
        if isinstance(value, bool):
            return value
        if value is not None:
            return str(value).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
    except Exception:
        logger.debug(
            "Could not inspect XGBoost CUDA build",
            exc_info=True,
        )
    return False


def detect_device(prefer_gpu: bool = True) -> DeviceInfo:
    cpu_count = os.cpu_count() or 1
    torch_available = False
    cuda_available = False
    mps_available = False
    accelerator = "cpu"
    name = platform.processor() or platform.machine() or "CPU"

    nvidia_driver_available, nvidia_device_names = (
        _detect_nvidia_driver()
    )
    xgboost_cuda_available = (
        _detect_xgboost_cuda()
        if prefer_gpu and nvidia_driver_available
        else False
    )

    try:
        import torch  # type: ignore

        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(
            mps_backend
            and mps_backend.is_available()
        )

        if prefer_gpu and cuda_available:
            accelerator = "cuda"
            name = torch.cuda.get_device_name(0)
        elif prefer_gpu and mps_available:
            accelerator = "mps"
            name = "Apple Metal Performance Shaders"
    except Exception:
        logger.debug(
            "PyTorch unavailable; using independent backend detection",
            exc_info=True,
        )

    if (
        prefer_gpu
        and nvidia_driver_available
        and not cuda_available
        and xgboost_cuda_available
    ):
        diagnostic = (
            "NVIDIA detectada. PyTorch no expone CUDA, pero "
            "XGBoost informa soporte CUDA; XGBoost puede usar GPU "
            "mientras TabNet seguira en CPU hasta corregir PyTorch."
        )
    elif (
        prefer_gpu
        and nvidia_driver_available
        and not cuda_available
    ):
        diagnostic = (
            "NVIDIA detectada por nvidia-smi, pero PyTorch no "
            "expone CUDA. Revise el build de torch y su "
            "compatibilidad con el driver."
        )
    elif prefer_gpu and cuda_available:
        diagnostic = "CUDA disponible para backends compatibles."
    elif prefer_gpu and mps_available:
        diagnostic = "MPS disponible para backends compatibles."
    else:
        diagnostic = "No se detecto acelerador utilizable; se usara CPU."

    info = DeviceInfo(
        accelerator=accelerator,
        name=name,
        cpu_count=cpu_count,
        torch_available=torch_available,
        cuda_available=cuda_available,
        mps_available=mps_available,
        nvidia_driver_available=nvidia_driver_available,
        nvidia_device_names=nvidia_device_names,
        xgboost_cuda_available=xgboost_cuda_available,
        diagnostic=diagnostic,
    )
    logger.info(
        "Device selected: %s | %s | CPU threads=%s | "
        "nvidia_driver=%s | nvidia_devices=%s | xgboost_cuda=%s | %s",
        info.accelerator,
        info.name,
        info.cpu_count,
        info.nvidia_driver_available,
        ", ".join(info.nvidia_device_names) or "none",
        info.xgboost_cuda_available,
        info.diagnostic,
    )
    return info


def xgboost_device_parameters(
    device: DeviceInfo,
) -> dict[str, str]:
    if device.xgboost_cuda_available:
        return {
            "device": "cuda",
            "tree_method": "hist",
        }
    return {
        "device": "cpu",
        "tree_method": "hist",
    }
