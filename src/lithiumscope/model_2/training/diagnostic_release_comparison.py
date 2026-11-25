"""Compatibility import; canonical implementation is lithiumscope.model_2.diagnostics.release_comparison."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.model_2.diagnostics.release_comparison")
