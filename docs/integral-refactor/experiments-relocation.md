# Experiment canonical relocation

Implemented first domain split, with legacy import compatibility. Remaining analysis, acquisition, reporting, georoc and validation scripts still require individual review; no experiment was executed or published.

- `analysis/run_vnext_m1.py` → `model_1/run_vnext_m1.py`
- `analysis/run_vnext_m1_diagnostics.py` → `model_1/run_vnext_m1_diagnostics.py`
- `analysis/validate_feot_expansion_m1.py` → `model_1/validate_feot_expansion_m1.py`
- `analysis/run_vnext_m2.py` → `model_2/run_vnext_m2.py`
- `analysis/run_vnext_m2_threshold.py` → `model_2/run_vnext_m2_threshold.py`
- `analysis/m2_generalization_pilot.py` → `model_2/m2_generalization_pilot.py`
- `analysis/m2_nested_calibration.py` → `model_2/m2_nested_calibration.py`
- `analysis/compare_vnext_trained_runs.py` → `integrated/compare_vnext_trained_runs.py`
- `analysis/verify_vnext_data_adequacy.py` → `integrated/verify_vnext_data_adequacy.py`
- `registry.py` → `shared/registry.py`
