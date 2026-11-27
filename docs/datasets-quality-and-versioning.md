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
records source coverage, totals and censoring evidence. Per-source raw_exclusions.csv
records each rejected source row position and its first rejection reason.

A candidate version contains dataset.csv, manifest.json, quality_report.json,
admitted.csv, quarantine.csv and excluded.csv. Per-source snapshots contain
harmonized dataset.csv, ingestion_audit.json and raw_exclusions.csv. All artifacts are hashed.
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

The full software validation menu runs python scripts/check_internal_imports.py automatically. The script can also be invoked independently. This static check scans
source, scripts and tests for retired dataset and analysis imports from the
relocation catalog. It does not import packages, resolve every dynamic or
relative import, run tests, or prove runtime behavior.

Use menu 5 > 3 > 1 for integrated software validation. It runs the retired-import check, syntax, dependency graph, code size, startup import smoke, Ruff, full pytest and coverage, and before/after selected protected-artifact hashes. Static failures do not suppress other checks. The full suite continues collecting runnable tests after collection errors, but such a run is never certified as passed. Summary JSON records each check and its log, pytest exit code, collection causes, and coverage. Exit 10 means incomplete evidence (including skips, missing coverage or unverifiable Git identity); nonzero pytest or failed checks mean failure. Missing Ruff is a failed check, not an automatic install. No production training or downloads are launched by the preflight; the test suite itself follows its existing fixtures and contracts. The additions include
tests for chemistry, coordinate precision, quality decisions, decoding, source
coverage, rollback, snapshot reuse, version comparison, provenance identity, raw exclusion accounting,
and M2 manifest evidence. These tests require local execution; no result is
asserted in this document.

## Canonical validation layout

Coordinate validation lives in `validation/spatial/coordinates.py`.
Header validation and target-column resolution share `validation/tabular/schema.py`.
Historical image-path resolution lives in `management/sources/model_2_paths.py`.
The validation root retains only its initializer, contracts, file-integrity checks,
and the cross-domain quality report. The retired modules are not compatibility shims.
The relocation catalog and static import check include these moves.

## Management ownership

M1 source resolution and acquisition share `management/sources/model_1.py`;
assembly is independent in `model_1_assembly.py`. M2 frozen and expanded
contracts share `model_2.py`, with linked imagery resolution in `model_2_paths.py`.
Image verification is opt-in at the manifest contract boundary and does not certify spectral validity.
No retired source modules are retained as compatibility shims.

## Source resolution and recovery

Preboot and local diagnostics resolve the configured cohort; expanded M2 is not silently replaced with frozen M2. Acquisition is injected by the caller and requires explicit authorization in diagnostics. Existing frozen hashes are not changed. Unfrozen M1 combined sources are reused only when source bytes, ingestion options, conservative dataset-code fingerprint and output hashes match. Candidate code fingerprints include dataset Python files and selected package versions; arbitrary external transformation dependencies are not automatically tracked.

Candidate CLI: `list` inventories versions, `verify <directory>` verifies their evidence, and `audit` / `georoc` accept `--recover-corrupt` to preserve a damaged matching version under `.quarantine` and rebuild from supplied inputs. Writers use per-version locks and fail clearly when another writer is active. A lock left by a process crash requires manual inspection and removal after ensuring no writer is active. Recovery is maintenance of candidate versions only; do not run recovery while other processes are reading that version. Image link checks verify availability and registered relocation hashes when requested, not spectral content. Full regression is menu 5 > 3 > 1.
