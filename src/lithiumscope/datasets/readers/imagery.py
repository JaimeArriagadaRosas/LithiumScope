"""Read raw multiband arrays without model or acquisition dependencies."""
from pathlib import Path
import numpy as np
from lithiumscope.core.exceptions import InputValidationError


def read_multiband_array(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError("GeoTIFF support requires the imagery extra: pip install -e '.[imagery]'") from exc
        with rasterio.open(path) as dataset:
            return dataset.read()
    raise InputValidationError(f"Unsupported imagery format: {path.suffix}")
