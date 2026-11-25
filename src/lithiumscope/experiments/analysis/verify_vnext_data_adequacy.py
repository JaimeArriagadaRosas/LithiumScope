"""Compatibility alias for :mod:`lithiumscope.experiments.integrated.verify_vnext_data_adequacy`."""
import sys
from importlib import import_module

if __name__ == "__main__":
    from runpy import run_module
    run_module("lithiumscope.experiments.integrated.verify_vnext_data_adequacy", run_name="__main__")
else:
    sys.modules[__name__] = import_module("lithiumscope.experiments.integrated.verify_vnext_data_adequacy")
