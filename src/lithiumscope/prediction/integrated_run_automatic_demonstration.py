"""Compatibility alias for :mod:`lithiumscope.prediction.demonstration.automatic`."""
import sys
from importlib import import_module

if __name__ == "__main__":
    from runpy import run_module
    run_module("lithiumscope.prediction.demonstration.automatic", run_name="__main__")
else:
    sys.modules[__name__] = import_module("lithiumscope.prediction.demonstration.automatic")
