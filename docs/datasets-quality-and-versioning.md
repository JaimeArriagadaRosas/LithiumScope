# Dataset quality, candidate versions and ingestion integrity

Branch scope: experimental datasets. The frozen M1/M2 inputs, hashes, production
configuration and trained models are not changed by these APIs.

## Explicit candidate workflow

From the repository root, with LithiumScope installed (or PYTHONPATH=src):

- python -m lithiumscope.datasets.management.candidates audit <input.csv> --model model_1
- python -m lithiumscope.datasets.management.candidates audit <manifest.csv> --model model_2
- python -m lithiumscope.datasets.management.candidates georoc <source1.csv> <source2.csv> --minimum-predictors 8
- python -m lithiumscope.datasets.management.candidates compare <old.csv> <new.csv> --identity source_dataset source_unique_id

The CLI writes immutable research artifacts under results/dataset_candidates.
No acquisition from the network or production promotion occurs. The GEOROC
workflow caches independent harmonizations and then combines their observations
without automatic cross-source deduplication. Its admission report covers the
harmonized rows, not every raw rejected row; the ingestion audit separately
records source coverage, totals and censoring evidence.

A candidate version contains dataset.csv, manifest.json, quality_report.json,
admitted.csv, quarantine.csv and excluded.csv. Per-source snapshots contain
harmonized dataset.csv and ingestion_audit.json. All artifacts are hashed.
The version identity includes source byte hashes, policy and an explicit
transformation revision. Change that revision when transformation logic changes.
A source-scoped audit observation ID identifies a row within exact source bytes;
it does not assert physical sample equivalence or persist across changed source
bytes. Cross-version comparison needs independently established stable sample
identifiers. Unknown analytical method, units, zeros, censoring, and M2
precision/CRS remain review concerns. Admission is not a release or independence
certificate. Image references are checked for presence by the quality contract;
actual patch bytes and spectral validity remain separate requirements.

## Existing operational paths

GEOROC ingestion now detects encoding, validates row widths and headers,
checks input hashes, fails on missing/incompatible sources by default and
records explicit skipped schemas when allowed. Data and evidence are staged
before publication, with rollback on ordinary errors. The combined M1 assembly
also records content hashes. Pair publication is not a cross-process lock or
an atomic filesystem-wide transaction; verify the evidence digest after an
interruption. Candidate directory publication avoids replacing prior versions.
The expanded M2 diagnostic path verifies structure and expansion evidence
when state_dir is configured. Frozen input verification is preserved.

The existing data manifest READY state describes artifact availability only,
not scientific suitability. Quality decisions and external overlap review
are explicit research outputs. Spatial proximity and shared references flag
review; the absence of detected overlap does not establish independence.

## Validation

Run python scripts/check_internal_imports.py first. This static check scans
source, scripts and tests for retired dataset and analysis imports from the
relocation catalog. It does not import packages, resolve every dynamic or
relative import, run tests, or prove runtime behavior.

Use menu option 5 for full regression and coverage. The additions include
tests for chemistry, coordinate precision, quality decisions, decoding, source
coverage, rollback, snapshot reuse, version comparison, provenance identity,
and M2 manifest evidence. These tests require local execution; no result is
asserted in this document.

## Deferred moves

The existing coordinates.py, paths.py and target_column.py remain in their
current locations until consumers have been fully audited. No compatibility
shims were restored. Structural migration of these paths should be performed
only as a coordinated change with all consumers and tests.
