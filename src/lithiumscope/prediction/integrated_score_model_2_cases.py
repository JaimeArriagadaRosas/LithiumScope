"""Compatibility alias for :mod:`lithiumscope.prediction.execution.model_2_scoring`."""
import sys
from importlib import import_module

if __name__ == "__main__":
    from runpy import run_module
    run_module("lithiumscope.prediction.execution.model_2_scoring", run_name="__main__")
else:
    sys.modules[__name__] = import_module("lithiumscope.prediction.execution.model_2_scoring")
