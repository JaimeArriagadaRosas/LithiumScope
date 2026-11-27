# Canonical Prediction and Reporting architecture

Prediction owns inputs, scientific analysis and integrated execution. Reporting owns PDF and Excel production, maps, charts, preview images and PDF reconstruction. Results owns atomic persistence; Persistence owns trusted artifact loading. Shared pure scientific report calculations live in core/scientific.

There are no compatibility aliases under Prediction. The relocation inventory records historical destinations. Use:
- `lithiumscope.prediction.execution.complete.run_complete_prediction`
- `lithiumscope.prediction.demonstration.automatic.run_automatic_demonstration`
- `python -m lithiumscope.reporting.rebuild_pdf --run <run-directory-or-id>`
- `python -m lithiumscope.evaluation.holdout`

Integrated outputs use typed frames, evidence and threshold contracts, atomic CSV/JSON writes, portable paths relative to the prediction session, hashes for published files and a publication status file. Historical report reconstruction uses saved evidence rather than training or inference. Report titles preserve experimental demonstration identity. The source manifest and imagery provenance must remain verifiable.

Architecture constraints: 300 lines per Python file, 20 direct Python files per directory; no Prediction aliases and no Reporting imports from model execution, Prediction or experiments. Logs use the existing session logger. No new workflow or training artifacts are introduced.

Regression verification must be run via the existing software validation menu. GitHub-only source editing does not prove execution or coverage.
