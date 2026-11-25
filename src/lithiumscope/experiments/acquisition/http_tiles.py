"""Compatibility import."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("lithiumscope.tools.acquisition.sentinel2.http_tiles")
