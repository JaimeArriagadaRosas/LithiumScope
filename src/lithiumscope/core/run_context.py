"""Compatibility import: canonical implementation is lithiumscope.core.execution.context."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.core.execution.context")
