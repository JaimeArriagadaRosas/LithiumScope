"""Compatibility import: canonical implementation is lithiumscope.datasets.validation.files."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.datasets.validation.files")
