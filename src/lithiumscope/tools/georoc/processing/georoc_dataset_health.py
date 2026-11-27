"""Compatibility import; canonical implementation lives in lithiumscope.datasets.validation.georoc_dataset_health."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("lithiumscope.datasets.validation.georoc.dataset_health")
