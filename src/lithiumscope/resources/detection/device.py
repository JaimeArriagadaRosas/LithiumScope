"""Backend-aware accelerator selection; CPU is always a valid fallback."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import platform

from lithiumscope.resources.compatibility.backends import probe_catboost_gpu, probe_xgboost_cuda
from lithiumscope.resources.detection.host import detect_gpus

logger = logging.getLogger("lithiumscope.resources.device")


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
    catboost_gpu_available: bool = False

    @property
    def is_gpu(self) -> bool:
        return self.accelerator in {"cuda", "mps"}

    @property
    def any_cuda_backend(self) -> bool:
        return self.cuda_available or self.xgboost_cuda_available or self.catboost_gpu_available


def detect_device(prefer_gpu: bool = True) -> DeviceInfo:
    gpus = detect_gpus()
    names = tuple(str(gpu["name"]) for gpu in gpus)
    xgb_ok = cat_ok = False
    notes = []
    if prefer_gpu and gpus:
        xgb, note = probe_xgboost_cuda()
        xgb_ok = xgb is True
        if note:
            notes.append(note)
        cat, note = probe_catboost_gpu()
        cat_ok = cat is True
        if note:
            notes.append(note)
    torch_available = cuda = mps = False
    accelerator = "cpu"
    name = platform.processor() or platform.machine() or "CPU"
    try:
        import torch
        torch_available = True
        cuda = bool(torch.cuda.is_available())
        backend = getattr(torch.backends, "mps", None)
        mps = bool(backend and backend.is_available())
        if prefer_gpu and cuda:
            accelerator, name = "cuda", torch.cuda.get_device_name(0)
        elif prefer_gpu and mps:
            accelerator, name = "mps", "Apple Metal Performance Shaders"
    except Exception:
        logger.debug("PyTorch unavailable", exc_info=True)
    if prefer_gpu and (cuda or mps or xgb_ok or cat_ok):
        notes.insert(0, "Aceleración disponible solo para los backends comprobados.")
    else:
        notes.insert(0, "No se detectó acelerador utilizable; se usará CPU.")
    info = DeviceInfo(accelerator, name, os.cpu_count() or 1, torch_available, cuda, mps,
                      bool(gpus), names, xgb_ok, " ".join(notes), cat_ok)
    logger.info("Device=%s torch_cuda=%s xgboost_cuda=%s catboost_gpu=%s", accelerator, cuda, xgb_ok, cat_ok)
    return info
