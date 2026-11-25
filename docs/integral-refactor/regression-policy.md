# Regression policy during refactoring

Every subsequent correction must identify the criterion, record a reproducible failing test or a stated expected invariant, make the smallest justified code change, and run the affected tests followed by regression. Historical batches in this branch intentionally deferred execution at the owner's instruction. They are **not** red/green evidence. The temporary option 5 → 4 executes syntax, lint, architecture, static imports and the full test suite for the exact local commit, and records nonpassing tests and protected artifact changes.

For follow-up repair, run the affected module(s) with `python scripts/run_tests.py -- tests/<component>/<test_file>.py -q`; then run `python scripts/run_tests.py --coverage -- -q`. Do not mark a criterion verified from the mere existence of a test module, a static source review, or a previous commit's results.

Do not fit production models, download sources, open holdout labels, promote candidates, or rewrite frozen artifacts as part of the regression. Synthetic fixtures and opt-in resource benchmarks are not real-data scientific evidence. Preserve failed and skipped outcomes. A successful full test suite does not certify scientific superiority or the external holdout.
