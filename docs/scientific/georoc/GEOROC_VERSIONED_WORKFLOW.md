# GEOROC research architecture and versioned execution

The two experimental acquisition packages are retired. Canonical acquisition
lives under tools/acquisition. GEOROC research is divided into screening,
provenance, imagery and specific studies; the shared protocol defines historical
locations. Reporting owns the evidence ledger; the unrelated UNAB source review
lives under experiments/validation.

## Local evidence pipeline

Run from the checkout with LithiumScope installed:

```
python -m lithiumscope.experiments.data_lab.source_studies.georoc.workflow --partition <part1.csv> --partition <part2.csv> --training <training.csv> --mamani <mamani.csv> --citations <citations.tab>
```

Preflight is read-only. Add `--execute` to create new evidence under
results/scientific_simulations/georoc_evidence_v1. The five stages are candidates,
screening, provenance, overlap and report. They share the immutable snapshot
manager; versions bind declared input files and code fingerprints, and can be
reused independently. Add `--recover` for explicit quarantine/rebuild of corrupt
snapshots. Raw partitions, training manifest and Mamani are never rewritten.
The summary supplies the `imagery_input` path. All evidence remains exploratory;
zero independent labels are approved by these stages.

## Registered exploratory imagery

```
python -m lithiumscope.experiments.model_2.external_validation.imagery.batch --source <imagery_input> --window 2025-01-01/2025-12-31 --limit 4
```

Add `--acquire` to authorize registered Sentinel acquisition and verified model
inference. The default is preflight only. The default limit of zero means all
eligible sites; an explicit positive limit bounds work. The new protocol uses
rounded coordinates as the patch center, stable site identities, the registered
local SCL policy and version-aware inference. These are different from the
historical pilot and cannot be treated as the same experiment. Results are under
results/scientific_simulations/georoc_registered_v1/<plan hash>. Plan hashes
bind candidate bytes, active model bytes, imagery config, code and selected
package versions. Per-site hashes bind patch, sidecar, SCL, scene audit and
scenario scores. Use `--recover` to preserve damaged site evidence before a
new attempt; no automatic deletion. Place STOP in the run directory to stop
between sites, remove it to resume. A crash may leave a lock requiring manual
inspection before removal. Registered provider uses its own bounded scene
search and GDAL HTTP timeouts; it does not impose a hard wall-clock timeout on
a complete site. Cache verification does not establish scientific independence.
A completed run can include abstentions. Failed sites make it incomplete.

Historical pilot and historical diagnostics remain executable by their
canonical paths, with original output conventions; their previous outputs
were not rewritten. Their historical acquisition policy is separate from the
new registered workflow. Explicit network access is required by the
single-sample and publication review scripts. Bibliography conflicts, primary
row ambiguity and duplicate evidence are preserved for manual review.
New stages and test fixtures do not certify original article methods or
geological surface correspondence.

## Verification

Menu 5 > 3 > 1 includes the extended retired-import scan and new tests.
The tests cover coordinate identity, ambiguous references, immutable stage
reuse, preflight, corrupt site recovery and publication authorization.
No live network or full experiment is required for those tests. Local
validation of the complete branch is still required.
