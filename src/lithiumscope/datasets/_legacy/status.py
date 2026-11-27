"""Compatibility import: canonical implementation is lithiumscope.datasets.management.status."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.datasets.management.status")
