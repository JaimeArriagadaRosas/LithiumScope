from __future__ import annotations
from pathlib import Path
import numpy as np
from lithiumscope.core.exceptions import InputValidationError

def load_imagery(path:Path)->np.ndarray:
    suffix=path.suffix.lower()
    if suffix==".npy": array=np.load(path)
    elif suffix in {".tif",".tiff"}:
        try: import rasterio
        except ImportError as exc: raise RuntimeError("GeoTIFF support requires the imagery extra: pip install -e '.[imagery]'") from exc
        with rasterio.open(path) as dataset: array=dataset.read()
    else: raise InputValidationError(f"Unsupported imagery format: {path.suffix}")
    array=np.asarray(array,dtype=np.float32)
    if array.ndim==2: array=array[None,:,:]
    if array.ndim!=3: raise InputValidationError(f"Expected imagery with shape [bands, height, width], got {array.shape}")
    return array
