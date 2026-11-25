"""Compatibility import for shared numerical metrics."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.core.scientific.classification_metrics")
