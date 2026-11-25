# Dataset management and provenance

- `lithiumscope.datasets.registry` → `lithiumscope.datasets.management.registry`
- `lithiumscope.datasets.status` → `lithiumscope.datasets.management.status`
- `lithiumscope.datasets.local_sources` → `lithiumscope.datasets.management.local_sources`
- `lithiumscope.datasets.model_1_sources` → `lithiumscope.datasets.management.model_1_sources`
- `lithiumscope.datasets.provisioner` → `lithiumscope.runtime.dataset_provisioning`
- `lithiumscope.datasets.manifest` → `lithiumscope.datasets.provenance.manifest`
- `lithiumscope.datasets.validator` → `lithiumscope.datasets.validation.files`

Model-specific provisioning now lives in runtime. Read-only frozen M2 validation stays in datasets/management. Linked-file manifests record ordered content identity and explicitly mark incomplete evidence. Compatibility aliases retain the historical module identity. Pending regression on local machine.
