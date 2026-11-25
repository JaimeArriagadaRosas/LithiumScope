"""Compatibility entrypoint; canonical experiment is lithiumscope.experiments.model_2.leakage_safe_m2."""
from importlib import import_module
import sys
if __name__=="__main__":
    from runpy import run_module
    run_module("lithiumscope.experiments.model_2.leakage_safe_m2",run_name="__main__")
else:
    sys.modules[__name__]=import_module("lithiumscope.experiments.model_2.leakage_safe_m2")
