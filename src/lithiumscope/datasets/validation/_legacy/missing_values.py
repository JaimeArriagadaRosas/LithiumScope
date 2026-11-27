"""Historical import; canonical implementation: lithiumscope.datasets.validation.tabular.missing_values."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.datasets.validation.tabular.missing_values")
