"""Compatibility alias for :mod:`lithiumscope.distribution.demonstration_bundle`."""
import sys
from importlib import import_module

if __name__ == "__main__":
    from runpy import run_module
    run_module("lithiumscope.distribution.demonstration_bundle", run_name="__main__")
else:
    sys.modules[__name__] = import_module("lithiumscope.distribution.demonstration_bundle")
