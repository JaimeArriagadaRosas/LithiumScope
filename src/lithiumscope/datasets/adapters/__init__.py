"""Source-specific dataset adapters."""
from .georoc import GeorocAdapterConfig, harmonize_georoc_frame

__all__ = ["GeorocAdapterConfig", "harmonize_georoc_frame"]

# Preserve old imports lazily, without importing acquisition or model code.
from pathlib import Path as _Path
__path__.append(str(_Path(__file__).with_name("_legacy")))
