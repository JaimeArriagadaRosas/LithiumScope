"""Compatibility import: canonical implementation is lithiumscope.tools.acquisition.georoc.query_contract."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.tools.acquisition.georoc.query_contract")
