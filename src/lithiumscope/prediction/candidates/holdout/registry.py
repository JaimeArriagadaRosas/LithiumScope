"""Compatibility import; canonical implementation is lithiumscope.evaluation.holdout.registry."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.evaluation.holdout.registry")
