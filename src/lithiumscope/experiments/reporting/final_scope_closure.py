"""Compatibility entrypoint for historical simulation synthesis."""
from importlib import import_module
import sys
if __name__=="__main__":
    from runpy import run_module
    run_module("lithiumscope.experiments.integrated.final_scope_closure",run_name="__main__")
else:
    sys.modules[__name__]=import_module("lithiumscope.experiments.integrated.final_scope_closure")
