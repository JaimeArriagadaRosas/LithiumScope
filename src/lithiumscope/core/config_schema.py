"""Compatibility import: canonical implementation is lithiumscope.core.configuration.schema."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.core.configuration.schema")
