"""Deserialize trusted model artifacts without keeping retired source modules.

Joblib stores Python class paths. Resolve only the two historical M1
class names at the serialization boundary; never install import aliases.
Do not use this loader for untrusted pickle/joblib files.
"""
from __future__ import annotations

from os import fspath

import joblib
from joblib import numpy_pickle

from lithiumscope.core.logging.session import get_logger

logger = get_logger("persistence.model_deserialization")

_RETIRED_M1_CLASSES = {
    ("lithiumscope.model_1.steps.step_08_preprocessing", "FeatureSchema"):
        "lithiumscope.model_1.training.preprocessing.tabular",
    ("lithiumscope.model_1.training.tabnet", "TabNetRegressorAdapter"):
        "lithiumscope.model_1.training.algorithms.tabnet",
}


class _Model1Unpickler(numpy_pickle.NumpyUnpickler):
    def find_class(self, module, name):
        canonical = _RETIRED_M1_CLASSES.get((module, name))
        if canonical is not None:
            logger.info("M1_ARTIFACT_CLASS_RELOCATED %s.%s -> %s.%s",
                        module, name, canonical, name)
            module = canonical
        return super().find_class(module, name)


def _load_model_1_stream(stream, filename):
    # Joblib >=1.5 and 1.4 have different stream/decompression interfaces.
    # Keep the adaptation local; do not mutate Joblib or sys.modules.
    if hasattr(numpy_pickle, "_validate_fileobject_and_memmap"):
        with numpy_pickle._validate_fileobject_and_memmap(stream, filename, None) as (reader, _):
            if isinstance(reader, str):
                raise ValueError("Pre-0.10 Joblib artifacts require explicit conversion")
            return _Model1Unpickler(filename, reader, ensure_native_byte_order=True).load()
    with numpy_pickle._read_fileobject(stream, filename, None) as reader:
        if isinstance(reader, str):
            raise ValueError("Pre-0.10 Joblib artifacts require explicit conversion")
        return _Model1Unpickler(filename, reader).load()


def load_model_artifact(source, *, model_group: str):
    """Load verified/trusted artifacts; M2 retains ordinary Joblib behavior."""
    if model_group != "model_1":
        return joblib.load(source)
    if hasattr(source, "read"):
        return _load_model_1_stream(source, getattr(source, "name", ""))
    with open(source, "rb") as stream:
        return _load_model_1_stream(stream, fspath(source))
