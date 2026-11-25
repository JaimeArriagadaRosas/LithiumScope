from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize(("old", "new"), [
    ("lithiumscope.prediction.report", "lithiumscope.prediction.exports.report_facade"),
    ("lithiumscope.prediction.rebuild_pdf", "lithiumscope.prediction.exports.rebuild_pdf"),
    ("lithiumscope.prediction.demo", "lithiumscope.prediction.demonstration.cases"),
    ("lithiumscope.prediction.imagery", "lithiumscope.prediction.inputs.imagery"),
    ("lithiumscope.prediction.integrated", "lithiumscope.prediction.execution.integrated"),
    ("lithiumscope.prediction.integrated_model_2_cases_from_input", "lithiumscope.prediction.inputs.model_2_cases"),
    ("lithiumscope.prediction.integrated_run_automatic_demonstration", "lithiumscope.prediction.demonstration.automatic"),
    ("lithiumscope.prediction.integrated_run_complete_prediction", "lithiumscope.prediction.execution.complete"),
    ("lithiumscope.prediction.integrated_score_model_2_cases", "lithiumscope.prediction.execution.model_2_scoring"),
    ("lithiumscope.prediction.integrated_write_outputs", "lithiumscope.prediction.exports.integrated_outputs"),
    ("lithiumscope.prediction.interpretation", "lithiumscope.prediction.analysis.interpretation"),
    ("lithiumscope.prediction.interpretation_summary", "lithiumscope.prediction.analysis.interpretation_summary"),
    ("lithiumscope.prediction.release_demo", "lithiumscope.distribution.demonstration_bundle"),
    ("lithiumscope.prediction.session", "lithiumscope.prediction.execution.session"),
])
def test_old_and_new_imports_are_same_module(old, new):
    assert import_module(old) is import_module(new)


def test_prediction_subpackages_are_present():
    root = Path(__file__).resolve().parents[2] / "src/lithiumscope/prediction"
    assert all((root / name / "__init__.py").is_file()
               for name in ("inputs", "execution", "analysis", "demonstration", "exports"))


def test_demo_publication_implementation_is_in_distribution():
    from lithiumscope.distribution.demonstration_bundle import build_demonstration_release_bundle
    assert build_demonstration_release_bundle.__module__ == "lithiumscope.distribution.demonstration_bundle"
