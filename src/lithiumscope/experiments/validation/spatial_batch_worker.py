"""Compatibility entrypoint; implementation in lithiumscope.tools.acquisition.sentinel2.spatial_batch_worker."""
from importlib import import_module
import sys
if __name__=="__main__":
    from runpy import run_module
    run_module("lithiumscope.tools.acquisition.sentinel2.spatial_batch_worker",run_name="__main__")
else:
    sys.modules[__name__]=import_module("lithiumscope.tools.acquisition.sentinel2.spatial_batch_worker")
