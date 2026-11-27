"""Historical import; canonical implementation: lithiumscope.datasets.adapters.georoc.ingestion."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.datasets.adapters.georoc.ingestion")
