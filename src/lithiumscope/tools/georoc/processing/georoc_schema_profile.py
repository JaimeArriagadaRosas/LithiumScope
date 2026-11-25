"""Compatibility import; canonical implementation lives in lithiumscope.datasets.validation.georoc_schema_profile."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("lithiumscope.datasets.validation.georoc_schema_profile")
