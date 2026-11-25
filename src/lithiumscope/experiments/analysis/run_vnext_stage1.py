"""Compatibility entrypoint; implementation in lithiumscope.experiments.model_1.run_vnext_stage1."""
from importlib import import_module
import sys
if __name__=="__main__":
    from runpy import run_module
    run_module("lithiumscope.experiments.model_1.run_vnext_stage1",run_name="__main__")
else:
    sys.modules[__name__]=import_module("lithiumscope.experiments.model_1.run_vnext_stage1")
