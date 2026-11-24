"""Compatibility imports for historical artifacts and external callers."""
from lithiumscope.resources.detection.device import DeviceInfo, detect_device
from lithiumscope.resources.compatibility.backends import xgboost_device_parameters

__all__ = ["DeviceInfo", "detect_device", "xgboost_device_parameters"]
