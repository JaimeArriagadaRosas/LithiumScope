"""Compatibility import: canonical implementation is lithiumscope.datasets.management.registry."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.datasets.management.registry")
