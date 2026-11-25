"""Compatibility import for historical protocol."""
from importlib import import_module
import sys
sys.modules[__name__]=import_module("lithiumscope.experiments.model_2.historical.vnext_experiments")
