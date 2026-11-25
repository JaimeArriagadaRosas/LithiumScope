"""Compatibility import; canonical implementation lives in lithiumscope.tools.acquisition.georoc.source_checkpoint."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("lithiumscope.tools.acquisition.georoc.source_checkpoint")
