"""Compatibility import; canonical implementation is lithiumscope.tools.acquisition.georoc.georoc_query_log."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("lithiumscope.tools.acquisition.georoc.georoc_query_log")
