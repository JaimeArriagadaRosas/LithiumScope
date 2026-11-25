"""Compatibility import; canonical implementation is lithiumscope.evaluation.holdout.scoring."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.evaluation.holdout.scoring")
