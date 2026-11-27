from __future__ import annotations
from pathlib import Path
import numpy as np
from lithiumscope.core.exceptions import InputValidationError
from lithiumscope.datasets.readers.imagery import read_multiband_array

def load_imagery(path:Path)->np.ndarray:
    array=read_multiband_array(path)
    array=np.asarray(array,dtype=np.float32)
    if array.ndim==2: array=array[None,:,:]
    if array.ndim!=3: raise InputValidationError(f"Expected imagery with shape [bands, height, width], got {array.shape}")
    return array
