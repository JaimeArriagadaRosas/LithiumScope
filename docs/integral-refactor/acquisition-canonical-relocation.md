# Canonical acquisition and adapters

Actual source implementations have been moved; legacy import shims preserve module identity and existing tests/clients.

- `lithiumscope.datasets.downloader` → `lithiumscope.tools.acquisition.tabular`
- `lithiumscope.datasets.georoc_query_contract` → `lithiumscope.tools.acquisition.georoc.query_contract`
- `lithiumscope.datasets.georoc_query_flow` → `lithiumscope.tools.acquisition.georoc.query_flow`
- `lithiumscope.datasets.georoc_filtered_acquisition` → `lithiumscope.tools.acquisition.georoc.filtered`
- `lithiumscope.tools.georoc.processing.georoc_material_parser` → `lithiumscope.datasets.adapters.georoc_material`
- `lithiumscope.datasets.georoc_ingestion_schema` → `lithiumscope.datasets.adapters.georoc_ingestion_schema`
- `lithiumscope.datasets.georoc_ingestion_transform` → `lithiumscope.datasets.adapters.georoc_ingestion_transform`
- `lithiumscope.datasets.georoc_ingestion` → `lithiumscope.datasets.adapters.georoc_ingestion`

Local regression is required; no criterion is closed by file relocation alone.
