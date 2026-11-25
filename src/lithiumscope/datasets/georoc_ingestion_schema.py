"""Compatibility import: canonical implementation is lithiumscope.datasets.adapters.georoc_ingestion_schema."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("lithiumscope.datasets.adapters.georoc_ingestion_schema")
