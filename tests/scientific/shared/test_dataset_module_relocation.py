"""Check historical module identity and the shared harmonization contract."""
from importlib import import_module

import pandas as pd
import pytest


@pytest.mark.parametrize("old,new", [
  [
    "lithiumscope.datasets.harmonization_deduplication",
    "lithiumscope.datasets.harmonization.deduplication"
  ],
  [
    "lithiumscope.tools.georoc.processing.georoc_normalizer",
    "lithiumscope.datasets.adapters.georoc_normalizer"
  ],
  [
    "lithiumscope.tools.georoc.processing.georoc_material_filter",
    "lithiumscope.datasets.adapters.georoc_material_filter"
  ],
  [
    "lithiumscope.tools.georoc.processing.georoc_raw_reader",
    "lithiumscope.datasets.readers.georoc_raw_reader"
  ],
  [
    "lithiumscope.tools.georoc.processing.georoc_schema_profile",
    "lithiumscope.datasets.validation.georoc_schema_profile"
  ],
  [
    "lithiumscope.tools.georoc.processing.georoc_dataset_health",
    "lithiumscope.datasets.validation.georoc_dataset_health"
  ],
  [
    "lithiumscope.tools.georoc.processing.georoc_postprocess",
    "lithiumscope.datasets.adapters.georoc_postprocess"
  ],
  [
    "lithiumscope.tools.georoc.processing.georoc_download_finalize",
    "lithiumscope.tools.acquisition.georoc.download_finalize"
  ],
  [
    "lithiumscope.tools.georoc.processing.georoc_source_checkpoint",
    "lithiumscope.tools.acquisition.georoc.source_checkpoint"
  ]
])
def test_legacy_module_identity(old, new):
    assert import_module(old) is import_module(new)


def test_harmonization_public_api():
    from lithiumscope.datasets.harmonization import canonicalize_coordinates

    result = canonicalize_coordinates(pd.DataFrame({"longitude": [-70.0], "latitude": [-23.0]}))
    assert result.loc[0, "Longitude"] == -70.0
    assert result.loc[0, "Latitude"] == -23.0


def test_shared_name_normalization():
    from lithiumscope.datasets.adapters.georoc_names import norm
    from lithiumscope.tools.georoc.query.georoc_query_html import norm as legacy

    assert legacy is norm
    assert norm("Li (ppm)") == "LIPPM"
