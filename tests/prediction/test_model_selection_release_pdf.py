"""Real LithiumScope menu PDF: selected models and provenance tables."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from lithiumscope.core.config import load_config
from lithiumscope.prediction.reporting.model_selection_evidence import (
    gather_selection_evidence,
)
from lithiumscope.prediction.reporting.sections.model_selection import (
    build_model_selection,
)
from lithiumscope.prediction.reporting.styles import build_styles
from lithiumscope.prediction.session import load_model_with_identity


def test_demonstration_pins_proposed_release_pair_without_activation():
    cfg = load_config("prediction")["demonstration"]["candidate_models"]
    assert cfg["model_1"] == "training_20261106_111653_m0300"
    assert cfg["model_2"] == "training_20260920_195933_m0300"


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


def test_selected_proposal_has_three_m1_and_all_eight_m2_runs():
    cfg = load_config("prediction")["demonstration"]["candidate_models"]
    id1 = load_model_with_identity("model_1",candidate_run_id=cfg["model_1"])[2]
    id2 = load_model_with_identity("model_2",candidate_run_id=cfg["model_2"])[2]
    report = gather_selection_evidence(id1,id2)
    assert len(report["model_1_runs"]) == 3
    assert len(report["model_2_runs"]) == 8
    assert len(report["model_2_170_six_algorithms"]) == 6
    assert report["model_1_paired"]["matched"] == 1004
    assert report["model_2_selected_run"] == id2["run_id"]
    assert report["model_2_reference_is_retrained_protocol_not_deployed_artifact"]
    assert not report["model_2_external_independent_holdout"]
    json.dumps(report,allow_nan=False)


def test_pdf_section_is_only_added_to_integrated_demonstration():
    cfg = load_config("prediction")["demonstration"]["candidate_models"]
    id1 = load_model_with_identity("model_1",candidate_run_id=cfg["model_1"])[2]
    id2 = load_model_with_identity("model_2",candidate_run_id=cfg["model_2"])[2]
    data = gather_selection_evidence(id1,id2)
    from types import SimpleNamespace
    context = SimpleNamespace(
        mode="demonstration",
        runtime={"model_selection_evidence":data},
    )
    story = build_model_selection(context,build_styles())
    assert len(story) > 20
    assert build_model_selection(
        SimpleNamespace(mode="complete",runtime=context.runtime),
        build_styles(),
    ) == []


def test_integrated_renderer_stays_original_and_has_selection_section():
    from inspect import getsource
    from lithiumscope.prediction.reporting.pdf import write_pdf_report
    source = getsource(write_pdf_report)
    assert "build_model_selection" in source
    assert "build_performance" in source
    assert "build_notes" in source
