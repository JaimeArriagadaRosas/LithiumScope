# M2 canonical architecture

Branch: `feat/training-refinement`.

## Ownership

- `model_2/data`: acquisition, resumable checkpoint state, deterministic candidate selection and source screening.
- `model_2/preparation`: feature construction, verified training-data cache and preparation of the training matrix.
- `model_2/eda`: exploratory analysis and descriptive artifacts, independent of the training orchestrator.
- `model_2/training`: competition orchestration, calibration, optimization, execution contracts and prospective validation.
- `model_2/evaluation`: metrics, ranking, selection, workbook outputs and saved OOF validation.
- `model_2/diagnostics`: diagnostics and recoverable evidence, including explicitly frozen historical comparisons.
- `model_2/methodology`: reusable scientific methods only.
- `experiments/model_2`: historical and exploratory studies, ablations, representation, external validation and closure experiments. No production M2 imports these experiments.

No compatibility packages are introduced. See `module-relocation.json` for moved entry points.
Existing historical evidence identities and scientific protocols are retained; historic report
comparisons explicitly reject incompatible cohorts rather than silently modifying them.
The generic saved OOF reader accepts new cohort sizes and fold counts, validates shared labels,
fold identities, probability bounds and sample identities when present.
Acquisition and evidence recovery use explicit paths; the historical reference identifiers
are intentionally fixed because they identify frozen scientific evidence.

## Regression checks

`tests/model_2/test_architecture.py` guards the 300-line / 20-direct-file constraints, absence
of retired packages, module aliases and reverse dependencies.
`tests/model_2/test_saved_oof.py` verifies arbitrary cohort sizes and invalid OOF rejection.
The M2 scientific, spectral, training, EDA and GEOROC tests remain the focused regression scope.
A full menu option 5 validation and real training are deferred until the overall refactor is complete.

Focused final regression: 283 passed, 5 skipped. Static audit: no unresolved local or literal imports, eager cycles, or retired references involving M2. The repository-wide audit still reports 15 retired references outside M2.
