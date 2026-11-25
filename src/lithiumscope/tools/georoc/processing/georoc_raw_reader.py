"""Compatibility import; canonical implementation lives in lithiumscope.datasets.readers.georoc_raw_reader."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("lithiumscope.datasets.readers.georoc_raw_reader")
