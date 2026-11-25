"""Bounded subprocess reads for reusable raster acquisition."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

class AssetTimeoutError(RuntimeError):
    """A COG asset did not complete within its bounded execution window."""


def read_asset_isolated(href: str, latitude: float, longitude: float, size_m: float,
                        pixels: int, *, mode: str = "reflectance",
                        timeout_seconds: int = 45) -> np.ndarray:
    """Isolate GDAL networking so a single blocked URL cannot stall the entire batch."""
    with tempfile.TemporaryDirectory(prefix="lithiumscope_s2_") as tmp:
        output = Path(tmp) / "asset.npy"
        command = [sys.executable, "-m", "lithiumscope.tools.acquisition.sentinel2.raster_worker",
                   "--href", href, "--lat", str(latitude), "--lon", str(longitude),
                   "--size", str(size_m), "--pixels", str(pixels), "--mode", mode,
                   "--output", str(output)]
        try:
            finished = subprocess.run(command, capture_output=True, text=True,
                                      timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AssetTimeoutError(f"{mode} raster read exceeded {timeout_seconds}s") from exc
        if finished.returncode != 0:
            detail = (finished.stderr or finished.stdout or "").strip()
            raise RuntimeError(f"{mode} raster read failed: {detail[-600:]}")
        if not output.is_file():
            raise RuntimeError(f"{mode} raster worker returned no data")
        array = np.load(output, allow_pickle=False)
        if array.shape != (pixels, pixels):
            raise ValueError(f"Unexpected band shape {array.shape}")
        return array


