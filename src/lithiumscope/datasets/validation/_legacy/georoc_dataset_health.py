"""Historical import; canonical implementation: lithiumscope.datasets.validation.georoc.dataset_health."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.datasets.validation.georoc.dataset_health")
