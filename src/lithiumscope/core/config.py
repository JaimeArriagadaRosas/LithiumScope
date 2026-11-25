"""Compatibility import: canonical implementation is lithiumscope.core.configuration.loading."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.core.configuration.loading")
