"""Compatibility import: canonical implementation is lithiumscope.datasets.provenance.manifest."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.datasets.provenance.manifest")
