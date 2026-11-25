"""Compatibility import; canonical implementation is lithiumscope.model_1.diagnostics.domain."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.model_1.diagnostics.domain")
