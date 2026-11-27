# Prediction and Reporting functional regression follow-up

The 07d101d software validation reported 1,236 passing, three failing and three skipped tests, with 56.65% repository line coverage. The 762db87 correction addresses the two retired scientific script imports and the invalid one-column CSV fixture. A new full execution is required to verify those corrections.

New regression scope:
- External regression and classification metrics: finite observations, insufficient truth, single-class evidence, threshold distinction, invalid scores.
- Offline Sentinel acquisition: per-case failures and provider-wide failure; demonstration source validation and overlap evidence.
- Integrated complete and demonstration modes: real scientific analysis, PNG figures/maps, PDF, Excel, manifest integrity and PDF reconstruction with controlled model outputs.
- Failed publication cannot be marked complete.
- EDA figure snapshots: source hashes, missing/altered evidence, path containment and copy integrity.
- Figure validity and cleanup on save failure; nonfinite chart observations are excluded before trend fitting.

Tests use temporary files and synthetic inputs. They do not train, promote or load actual model artifacts or contact Sentinel services. Existing scientific rules and protected artifacts are unchanged. No new workflows are required. Coverage is a target, not an observed result until the existing software validation is executed.
