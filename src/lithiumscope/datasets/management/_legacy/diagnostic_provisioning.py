"""Historical import; canonical implementation: lithiumscope.datasets.management.sources.diagnostic."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.datasets.management.sources.diagnostic")
