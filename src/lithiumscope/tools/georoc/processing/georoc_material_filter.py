"""Compatibility import; canonical implementation lives in lithiumscope.datasets.adapters.georoc_material_filter."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("lithiumscope.datasets.adapters.georoc_material_filter")
