"""Compatibility alias for :mod:`lithiumscope.experiments.model_1.run_vnext_m1_diagnostics`."""
import sys
from importlib import import_module

if __name__ == "__main__":
    from runpy import run_module
    run_module("lithiumscope.experiments.model_1.run_vnext_m1_diagnostics", run_name="__main__")
else:
    sys.modules[__name__] = import_module("lithiumscope.experiments.model_1.run_vnext_m1_diagnostics")
