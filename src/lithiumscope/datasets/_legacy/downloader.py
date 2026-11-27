"""Compatibility import: canonical implementation is lithiumscope.tools.acquisition.tabular."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.tools.acquisition.tabular")
