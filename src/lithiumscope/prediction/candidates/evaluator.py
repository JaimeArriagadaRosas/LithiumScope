"""Compatibility import; canonical implementation is lithiumscope.evaluation.candidates.evaluator."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.evaluation.candidates.evaluator")
