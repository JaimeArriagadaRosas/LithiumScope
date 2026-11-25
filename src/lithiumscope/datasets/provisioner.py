"""Compatibility import: canonical implementation is lithiumscope.runtime.dataset_provisioning."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.runtime.dataset_provisioning")
