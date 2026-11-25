"""Compatibility entrypoint; canonical study is lithiumscope.model_1.methodology.dataset_comparison."""
import sys
from importlib import import_module

if __name__ == "__main__":
    from runpy import run_module
    run_module("lithiumscope.model_1.methodology.dataset_comparison",run_name="__main__")
else:
    sys.modules[__name__]=import_module("lithiumscope.model_1.methodology.dataset_comparison")
