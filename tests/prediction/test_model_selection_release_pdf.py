"""Real LithiumScope menu PDF: selected models and provenance tables."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.prediction.analysis.model_selection_evidence import (
    gather_selection_evidence, required_historical_paths, M1_RUNS, M2_RUNS,
)
from lithiumscope.reporting.sections.model_selection import (
    build_model_selection,
)
from lithiumscope.reporting.styles import build_styles
from lithiumscope.prediction.session import load_model_with_identity


# These optional integration tests require the locally archived release
# artifacts, which are deliberately excluded from source-control checkouts.
# Lightweight dispatch and renderer-structure tests still execute in CI.
LOCAL_RELEASE_ARTIFACTS = all(
    (PROJECT_ROOT / "models" / group / "trained" / run / "model.joblib").is_file()
    for group, run in (
        ("model_1", "training_20261106_111653_m0300"),
        ("model_2", "training_20260920_195933_m0300"),
    )
)
requires_local_release = pytest.mark.skipif(
    not LOCAL_RELEASE_ARTIFACTS,
    reason="Requires local versioned .joblib files excluded from source control",
)


def test_demonstration_defaults_to_new_local_candidates_without_activation():
    cfg = load_config("prediction")["demonstration"]["candidate_models"]
    assert cfg["model_1"] == "latest_local_candidate"
    assert cfg["model_2"] == "latest_local_candidate"


@requires_local_release
def test_current_promoted_release_is_accepted_as_explicit_demo_pin():
    bundle, model_path, ident = load_model_with_identity(
        "model_2",
        candidate_run_id="training_20260920_195933_m0300",
    )
    assert model_path.is_file()
    assert len(bundle["features"]) == 126
    assert ident["candidate_status"] == "promoted"
    assert ident["selection"] == "pinned_published_release_for_demonstration"
    assert ident["release_promoted_by_demonstration"] is False


def _assert_complete_history(report, id2):
    assert report["state"] == "HISTORICAL_COMPARISON"
    assert len(report["model_1_runs"]) == 3
    assert len(report["model_2_runs"]) == 8
    assert len(report["model_2_170_six_algorithms"]) == 6
    assert report["model_1_paired"]["matched"] == 1004
    assert report["model_2_selected_run"] == id2["run_id"]
    assert report["model_2_reference_is_retrained_protocol_not_deployed_artifact"]
    assert not report["model_2_external_independent_holdout"]
    json.dumps(report, allow_nan=False)


def _assert_missing_history(report, missing):
    assert report["state"] == "HISTORICAL_EVIDENCE_UNAVAILABLE"
    assert report["error_type"] == "FileNotFoundError"
    assert set(report["missing_files"]) == set(missing)
    assert "model_1_runs" not in report and "model_2_runs" not in report
    assert not report["model_2_external_independent_holdout"]
    assert report["user_release_approval_required"]


@requires_local_release
def test_selected_proposal_has_three_m1_and_all_eight_m2_runs():
    id1 = load_model_with_identity("model_1",candidate_run_id=M1_RUNS[2])[2]
    id2 = load_model_with_identity("model_2",candidate_run_id=M2_RUNS[0])[2]
    report = gather_selection_evidence(id1,id2)
    missing = [p.as_posix() for p in required_historical_paths() if not (PROJECT_ROOT/p).is_file()]
    if missing:
        _assert_missing_history(report,missing)
    else:
        _assert_complete_history(report,id2)


def _assert_renderer(data, *, complete):
    from types import SimpleNamespace
    context = SimpleNamespace(mode="demonstration",runtime={"model_selection_evidence":data})
    story = build_model_selection(context,build_styles())
    if complete:
        assert len(story) > 20
    else:
        assert len(story) == 3
        assert "no disponible" in story[1].getPlainText()
    assert build_model_selection(SimpleNamespace(mode="complete",runtime=context.runtime),
                                 build_styles()) == []


@requires_local_release
def test_pdf_section_is_only_added_to_integrated_demonstration():
    id1 = load_model_with_identity("model_1",candidate_run_id=M1_RUNS[2])[2]
    id2 = load_model_with_identity("model_2",candidate_run_id=M2_RUNS[0])[2]
    data = gather_selection_evidence(id1,id2)
    missing = [p.as_posix() for p in required_historical_paths() if not (PROJECT_ROOT/p).is_file()]
    if missing:
        _assert_missing_history(data,missing)
    else:
        _assert_complete_history(data,id2)
    _assert_renderer(data,complete=not missing)


@pytest.fixture
def synthetic_history(tmp_path):
    """Synthetic test-only archive. Never write or invent actual project evidence."""
    import pandas as pd

    def write(relative, value):
        path=tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding="utf-8")

    for group,runs in (("model_1",M1_RUNS),("model_2",M2_RUNS)):
        for run in runs:
            folder=Path("models")/group/"trained"/run
            write(folder/"metadata.json",{"algorithm":"svm" if group=="model_1" else "random_forest",
                "samples":42,"metrics":{"rmse":2.,"mae":1.}})
            write(folder/"candidate_status.json",{"status":"pending"})
    errors={"old":{"rmse":4.,"mae":2.},"new":{"rmse":3.,"mae":1.}}
    write("results/scientific_simulations/vnext_20261106/stage5_training_comparison/m1_paired_comparison.json",
          {"comparison":{"matched":1004,"different_outer_fold":3},
           "by_subset":{"all_shared":errors,"high_lithium_q75_common":errors},
           "cluster_bootstrap":{},"development_improvement_gate":{"release_certified":False}})
    write("results/scientific_simulations/m2_rededge_v4_20261113/cohort_706/spectral_identity_summary.json",
          {"distinct_exact_signatures_170":20,"excess_redundant_rows":2,
           "mixed_fixed_class_signatures":1,"minimum_classification_errors_due_only_to_identical_X":1})
    tables=tmp_path/"results/model_2/runs"/M2_RUNS[-1]/"tables"
    tables.mkdir(parents=True)
    pd.DataFrame([{"algorithm":alg,"roc_auc":auc,"average_precision":.3,"brier":.2}
                  for alg,auc in [("original_release_protocol_retrained",.5),("random_forest",.6)]
                 ]).to_csv(tables/"old_release_protocol_comparison_706.csv",index=False)
    pd.DataFrame([{"algorithm":"random_forest","auc_delta_ci95_low":-.1,"auc_delta_ci95_high":.2,
                   "ap_delta_ci95_low":-.1,"ap_delta_ci95_high":.1}
                 ]).to_csv(tables/"old_release_protocol_intervals_706.csv",index=False)
    pd.DataFrame([{"algorithm":alg,"status":"ok","roc_auc_mean":.6,"average_precision_mean":.3}
                  for alg in ("random_forest","extra_trees","hist_gradient_boosting","xgboost","catboost","svm_rbf")
                 ]).to_csv(tables/"competition_ranking.csv",index=False)
    return tmp_path


def test_complete_historical_annex_without_local_release(synthetic_history):
    id1,id2={"run_id":M1_RUNS[2]},{"run_id":M2_RUNS[0]}
    report=gather_selection_evidence(id1,id2,root=synthetic_history)
    _assert_complete_history(report,id2)
    _assert_renderer(report,complete=True)


def test_partial_historical_annex_never_invents_metrics(synthetic_history):
    missing=required_historical_paths()[-1]
    (synthetic_history/missing).unlink()
    data=gather_selection_evidence({"run_id":M1_RUNS[2]},{"run_id":M2_RUNS[0]},root=synthetic_history)
    _assert_missing_history(data,[missing.as_posix()])
    _assert_renderer(data,complete=False)


def test_integrated_renderer_stays_original_and_has_selection_section():
    from inspect import getsource
    from lithiumscope.reporting.pdf import write_pdf_report
    source = getsource(write_pdf_report)
    assert "build_model_selection" in source
    assert "build_performance" in source
    assert "build_notes" in source



@pytest.mark.parametrize(
    "status, selection",
    [
        ("promoted", "pinned_published_release_for_demonstration"),
        ("pending", "experimental_demonstration"),
    ],
)
def test_versioned_demo_loads_synthetic_model_without_local_artifacts(
    tmp_path, monkeypatch, status, selection
):
    """CI verifies both permitted states even without historical .joblib files."""
    from lithiumscope.prediction import session

    root = tmp_path / "models"
    run_id = "training_20261106_111653_m0300"
    directory = root / "model_2" / "trained" / run_id
    directory.mkdir(parents=True)
    joblib.dump(
        {"algorithm": "catboost", "features": [f"feature_{i}" for i in range(126)]},
        directory / "model.joblib",
    )
    (directory / "candidate_status.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    (directory / "metadata.json").write_text(
        json.dumps({"run_id": run_id, "algorithm": "catboost"}), encoding="utf-8"
    )
    monkeypatch.setattr(session, "MODELS_DIR", root)
    bundle, path, ident = session.load_model_with_identity(
        "model_2", candidate_run_id=run_id
    )
    assert path.is_file() and len(bundle["features"]) == 126
    assert ident["candidate_status"] == status
    assert ident["selection"] == selection
    assert ident["release_promoted_by_demonstration"] is False
    assert json.loads((directory / "candidate_status.json").read_text()) == {
        "status": status
    }
