from __future__ import annotations

from dataclasses import dataclass
import os
import platform

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

    @property
    def is_gpu(self) -> bool:
        return self.accelerator in {"cuda", "mps"}


def detect_device(prefer_gpu: bool = True) -> DeviceInfo:
    cpu_count = os.cpu_count() or 1
    torch_available = False
    cuda_available = False
    mps_available = False
    accelerator = "cpu"
    name = platform.processor() or platform.machine() or "CPU"

    try:
        import torch  # type: ignore

        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())

        if prefer_gpu and cuda_available:
            accelerator = "cuda"
            name = torch.cuda.get_device_name(0)
        elif prefer_gpu and mps_available:
            accelerator = "mps"
            name = "Apple Metal Performance Shaders"
    except Exception:
        logger.debug("PyTorch unavailable; using CPU detection only", exc_info=True)

    info = DeviceInfo(
        accelerator=accelerator,
        name=name,
        cpu_count=cpu_count,
        torch_available=torch_available,
        cuda_available=cuda_available,
        mps_available=mps_available,
    )
    logger.info(
        "Device selected: %s | %s | CPU threads=%s",
        info.accelerator,
        info.name,
        info.cpu_count,
    )
    return info


def xgboost_device_parameters(device: DeviceInfo) -> dict[str, str]:
    if device.accelerator == "cuda":
        return {"device": "cuda", "tree_method": "hist"}
    return {"device": "cpu", "tree_method": "hist"}
