# Test architecture

Tests are classified by the primary purpose of their assertions, not by the package under test. Every test module belongs to exactly one primary category:

| Category | Purpose |
| --- | --- |
| architecture | Boundaries, dependencies, contracts and structure |
| unit | Isolated behavior |
| integration | Component interaction |
| functional | User-visible contracts |
| scientific | Methodology and evidence |
| regression | Historical behavior and reproducibility |
| performance | Resource and execution behavior |
| reliability | Persistence, logging and validation |

The original 296 modules have been relocated to canonical paths; the historical map is at `docs/integral-refactor/test-relocation-purpose.json`. New tests use the same four-part layout `tests/<category>/<responsibility>/test_*.py`. The active criterion catalog is `docs/integral-refactor/criterion-tests.json`.

## Fixtures and selection

Shared fixtures are registered in `tests/conftest.py`. The `tests/fixtures/` directory is reserved for reusable fixture inputs, not executable suites. Synthetic inputs must not be presented as external scientific evidence.

The collection hook in `scripts/pytest_classification.py` assigns exactly one primary category before standard pytest marker filtering. Transversal markers can overlap: `m1`, `m2`, `eda`, `training`, `slow`, `local_artifacts`. The canonical scope registry is `docs/integral-refactor/test-marker-scopes.json`; it starts from model-specific component provenance and explicit test/module names. Explicit decorators document additional scope, slow execution and local evidence requirements. Scope tags are conservative, not an exhaustive coverage claim; absence of a model marker is not proof of model independence.

Default execution remains the entire suite. No marker is automatically excluded:

```sh
python -m pytest -q
python -m pytest -q -m architecture
python -m pytest -q -m "scientific and m2"
python scripts/run_tests.py --coverage -- -q
```

JUnit records the canonical relative path, primary category and marker set for collected tests. The criterion evaluator prefers this path and does not substitute a similar basename when an explicit path is unknown. Historical JUnit without these properties is supported by the existing unique-module matching policy. Associations do not close scientific criteria.

Limits: 300 lines per Python file, at most 20 direct Python files per directory, unique test module basenames. Collection and execution must be validated separately from static checks.
