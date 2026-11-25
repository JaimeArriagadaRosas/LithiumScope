"""Compatibility entrypoint; canonical implementation is lithiumscope.prediction.exports.report_facade."""
from importlib import import_module
import sys
sys.modules[__name__]=import_module("lithiumscope.prediction.exports.report_facade")
