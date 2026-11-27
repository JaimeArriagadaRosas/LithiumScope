"""Select adapters by the existing dataset keys, without implicit acquisition."""
from lithiumscope.datasets.management.registry import get_dataset_spec

def get_adapter(key):
    get_dataset_spec(key)
    if key == "mamani09_public_mirror":
        from .mamani09 import adapt_mamani09
        return adapt_mamani09
    if key == "georoc_andean_arc":
        from .georoc.ingestion import ingest_georoc_files
        return ingest_georoc_files
    if key == "fregeneda_almendra":
        from .spectral.fregeneda_almendra import adapt_fregeneda
        return adapt_fregeneda
    if key == "greenpeg":
        from .spectral.greenpeg import adapt_greenpeg
        return adapt_greenpeg
    if key == "sentinel2":
        from .sentinel2 import inspect_sentinel_patch
        return inspect_sentinel_patch
    raise ValueError(f"No adapter registered for {key}")
