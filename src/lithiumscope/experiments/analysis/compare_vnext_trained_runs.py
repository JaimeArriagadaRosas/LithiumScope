"""Compatibility alias for :mod:`lithiumscope.experiments.integrated.compare_vnext_trained_runs`."""
import sys
from importlib import import_module

if __name__ == "__main__":
    from runpy import run_module
    run_module("lithiumscope.experiments.integrated.compare_vnext_trained_runs", run_name="__main__")
else:
    sys.modules[__name__] = import_module("lithiumscope.experiments.integrated.compare_vnext_trained_runs")
