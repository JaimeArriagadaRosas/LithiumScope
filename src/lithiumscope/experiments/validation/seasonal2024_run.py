"""Compatibility entrypoint; canonical experiment is lithiumscope.experiments.model_2.historical.seasonal2024_run."""
from importlib import import_module
import sys
if __name__=="__main__":
    from runpy import run_module
    run_module("lithiumscope.experiments.model_2.historical.seasonal2024_run",run_name="__main__")
else:
    sys.modules[__name__]=import_module("lithiumscope.experiments.model_2.historical.seasonal2024_run")
