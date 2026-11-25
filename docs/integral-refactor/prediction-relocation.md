# Prediction canonical relocation

Implementation moved, old module names resolve to the same module object for tests, notebooks and compatibility. No algorithm or historical release changed. Pending local regression and PDF inspection.

- `prediction/demo.py` → `prediction/demonstration/cases.py`
- `prediction/imagery.py` → `prediction/inputs/imagery.py`
- `prediction/integrated.py` → `prediction/execution/integrated.py`
- `prediction/integrated_model_2_cases_from_input.py` → `prediction/inputs/model_2_cases.py`
- `prediction/integrated_run_automatic_demonstration.py` → `prediction/demonstration/automatic.py`
- `prediction/integrated_run_complete_prediction.py` → `prediction/execution/complete.py`
- `prediction/integrated_score_model_2_cases.py` → `prediction/execution/model_2_scoring.py`
- `prediction/integrated_write_outputs.py` → `prediction/exports/integrated_outputs.py`
- `prediction/interpretation.py` → `prediction/analysis/interpretation.py`
- `prediction/interpretation_summary.py` → `prediction/analysis/interpretation_summary.py`
- `prediction/release_demo.py` → `distribution/demonstration_bundle.py`
- `prediction/session.py` → `prediction/execution/session.py`
