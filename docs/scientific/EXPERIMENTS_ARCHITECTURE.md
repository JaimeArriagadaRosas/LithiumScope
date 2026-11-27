# Experimental architecture

## Responsibilities

- `lithiumscope/model_1/eda` and `lithiumscope/model_2/eda`: canonical descriptive EDA for the respective operational prepared inputs.
- `results/eda_artifacts.py`: evidence manifests and integrity checks; `results/eda_cache.py`: current input/code/configuration index. Descriptive evidence cannot become a fitted training transform.
- `experiments/data_lab`: candidate source research, not a second implementation of the models' EDA. `screening` contains cross-source screening and its historical cohort audit; `source_studies` contains source-specific research.
- `experiments/model_1`: experimental protocols, treatments, data expansion, methods, evaluation and studies of concentration estimation.
- `experiments/model_2`: experimental representation, calibration, spatial and temporal evaluation, external validation and studies of prospectivity.
- `experiments/generalization`, `shared`, `scenarios`, `integrated`, `reporting` and `validation`: cross-model experimental concerns.

The source study registry describes implemented research entry points and model applicability. Only the existing GEOROC candidate EDA for M1 runs automatically as part of the laboratory analysis. The M2 GEOROC external evidence pipeline is an explicit operation with declared inputs; it does not automatically acquire satellite imagery. Spectral libraries have operational adapters but are not yet registered as experiments until their scientific study is implemented.

## Canonical entry points

```text
python -m lithiumscope.experiments.data_lab.source_studies.georoc.workflow --help
python -m lithiumscope.experiments.model_2.external_validation.imagery.batch --help
python -m lithiumscope.experiments.data_lab.source_studies.chen
python -m lithiumscope.experiments.model_1.studies.balanced.academic --help
python -m lithiumscope.experiments.model_2.studies.grouped_training
```

The last two commands may fit experimental models and are not software-validation commands. Consult source arguments and prerequisites before invoking historical experiments. Original source hashes, scientific constants, archived run identifiers and historical output names are preserved. The new registered imagery workflow is distinct from historical exploratory acquisition.

Chen descriptive profiling, Meixner workbook interpretation, directed whole-rock interpretation and shared raw measurement parsing are under source_studies. Audits requiring a specific M1 training cohort remain in model_1/data_expansion. The OASIS-like cohort selection is owned by model_1/protocols/cohorts.py, separate from its fitting runner. Existing historical treatment and training study calculations are retained.

## EDA cache

Use the canonical EDA index for reuse from the laboratory and experimental preparation. A prepared data snapshot's historical EDA path is not authoritative. The index binds source, configuration, upstream preparation, code and dependency identities and checks the evidence owner, source hash and inventory. Cache invalidation must not alter prepared snapshots or frozen input artifacts.

## Validation

Run the software validation menu 5 > 3 > 1. The retired-import check also resolves relative imports inside src and checks the retired experimental package prefixes. Tests cover canonical owners, source study isolation, EDA cache corruption, model-specific source registration and existing experimental behavior. Local execution is required to confirm runtime imports and regression behavior; the Git tree inventory and relocation catalog alone cannot prove that the test suite passes.
