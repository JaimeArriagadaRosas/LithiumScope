"""Compatibility import; canonical implementation is lithiumscope.core.reproducibility.hashing."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.core.reproducibility.hashing")
