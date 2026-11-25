"""Compatibility import; canonical implementation is lithiumscope.prediction.execution.candidate_scoring."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.prediction.execution.candidate_scoring")
