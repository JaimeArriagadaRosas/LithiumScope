# Experiments analysis migration

The redundant experiments/analysis package was removed. Use canonical modules under experiments/model_1, experiments/model_2, experiments/integrated, and model_1/methodology/qc_protocols. Historical M2 protocols remain under experiments/model_2/historical.

Saved-run comparisons live in experiments/integrated/comparison/compare_m1.py and compare_m2.py, with evidence.py holding the unchanged metrics and OOF checks. The integrated runner coordinates both comparisons and records a structured operation log. An explicit root controls default output placement; an explicit output overrides it. The default root is resolved by core.paths, including LITHIUMSCOPE_HOME.

No training or model promotion is performed by this comparison. Scientific thresholds and calculations were retained. Full repository consumer verification and runtime tests remain pending; the default-branch search does not establish branch-wide reference coverage.
