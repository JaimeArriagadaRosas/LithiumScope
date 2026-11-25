"""Compatibility import; canonical implementation is lithiumscope.evaluation.candidates.gates."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.evaluation.candidates.gates")
