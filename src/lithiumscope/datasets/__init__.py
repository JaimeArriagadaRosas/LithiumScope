"""Dataset contracts, preparation and provenance."""

# Preserve old imports lazily, without importing acquisition or model code.
from pathlib import Path as _Path
__path__.append(str(_Path(__file__).with_name("_legacy")))
