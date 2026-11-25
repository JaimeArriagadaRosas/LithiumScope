"""Compatibility entrypoint; canonical implementation is lithiumscope.prediction.exports.rebuild_pdf."""
from importlib import import_module
import sys
if __name__=="__main__":
    from runpy import run_module
    run_module("lithiumscope.prediction.exports.rebuild_pdf",run_name="__main__")
else:
    sys.modules[__name__]=import_module("lithiumscope.prediction.exports.rebuild_pdf")
