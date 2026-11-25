"""Compatibility import; canonical implementation is lithiumscope.model_1.diagnostics.group_intervals."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.model_1.diagnostics.group_intervals")
