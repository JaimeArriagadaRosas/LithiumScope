"""M1 -> three -> M2 -> one: report-only reproducible decision contracts."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, PageBreak

from lithiumscope.prediction.reporting.decision import calculate_decision
from lithiumscope.prediction.reporting.sections.decision import build_decision_page
from lithiumscope.prediction.reporting.sections.spatial import build_spatial
from lithiumscope.prediction.reporting.styles import build_styles


def _cases():
    return pd.DataFrame({
        "case_id": [f"site-{i}" for i in range(10)],
        "Li_icpms_predicted": [11, 19, 30, 22, 8, 7, 3, 18, 21, 10],
        "Li_icpms_interval_low_q90": [1, 9, 20, 12, 0, 0, 0, 8, 11, 0],
        "Li_icpms": [100, 0, 5, 25, 0, 0, 0, 0, 0, 0],
        "prospectivity_score": [.2, .1, .3, .8, .7, .6, .5, .4, .9, .15],
        "model_1_applicability_warning": ["OK"] * 10,
        "model_2_applicability_warning": ["OK"] * 10,
    })


def test_independent_model_predictions_reduced_only_in_renderer():
    result = calculate_decision(_cases())
    assert set(result.selected.case_id) == {"site-2", "site-3", "site-8"}
    assert result.winner_id == "site-8"
    assert result.status == "COMPARABLE"
    assert not result.retrospective["m2_winner_matches_observed"]
    assert result.m1_rule.startswith("cota inferior q90")


def test_true_lithium_never_controls_selection():
    source = _cases()
    first = calculate_decision(source)
    source["Li_icpms"] = np.arange(10, dtype=float) * 500
    second = calculate_decision(source)
    assert first.winner_id == second.winner_id
    assert first.selected.case_id.tolist() == second.selected.case_id.tolist()


def test_bad_m2_quality_blocks_highest_scoring_candidate():
    source = _cases()
    source.loc[source.case_id == "site-8", "model_2_applicability_warning"] = "OUT_OF_DOMAIN"
    result = calculate_decision(source)
    assert result.winner_id == "site-3"
    assert result.status == "PROVISIONAL"


def test_insufficient_m1_applicability_abstains():
    source = _cases()
    source["model_1_applicability_warning"] = "OUT_OF_DOMAIN"
    source.loc[:1, "model_1_applicability_warning"] = "OK"
    result = calculate_decision(source)
    assert result.winner_id is None
    assert result.selected.empty
    assert result.status == "INSUFFICIENT"


def test_m2_tie_or_insufficient_data_abstains():
    source = _cases()
    source.loc[source.case_id == "site-8", "prospectivity_score"] = .8
    assert calculate_decision(source).winner_id is None
    source.loc[source.case_id.isin(["site-2", "site-3"]), "model_2_applicability_warning"] = "OUT_OF_DOMAIN"
    assert calculate_decision(source).status == "ABSTAIN"


def test_missing_q90_switches_all_candidates_to_same_fallback():
    source = _cases().drop(columns="Li_icpms_interval_low_q90")
    result = calculate_decision(source)
    assert set(result.selected.case_id) == {"site-2", "site-3", "site-8"}
    assert "Li estimado M1" in result.m1_rule


def test_rendered_page_is_dynamic_and_exactly_page_two(tmp_path):
    PdfReader = pytest.importorskip("pypdf").PdfReader

    def render(data, target):
        context = SimpleNamespace(
            title="LithiumScope - Demo",
            run_id="fixture",
            mode="demonstration",
            interpretation=(
                "INTERPRETACIÓN CIENTÍFICA INTEGRADA\n"
                "Pregunta del Modelo 1: datos\n"
                "Pregunta del Modelo 2: imágenes\n"
                "Cobertura de la ejecución:\n"
                "- Casos evaluados por Modelo 1: 10"
            ),
            figures=(),
            maps=(),
            paired=data,
        )
        story = build_spatial(context, build_styles())
        assert len([x for x in story if isinstance(x, PageBreak)]) == 2
        doc = SimpleDocTemplate(
            str(target), pagesize=A4, leftMargin=44,
            rightMargin=44, topMargin=44, bottomMargin=45,
        )
        doc.build(story)
        pages = PdfReader(str(target)).pages
        assert len(pages) >= 3
        assert "Decisión integrada" in pages[1].extract_text()
        assert "Cobertura de la ejecución:" in pages[2].extract_text()
        assert "Cobertura de la ejecución:" not in pages[1].extract_text()
        return pages[1].extract_text()

    first = render(_cases(), tmp_path / "first.pdf")
    changed = _cases()
    changed.loc[changed.case_id == "site-8", "prospectivity_score"] = .01
    second = render(changed, tmp_path / "second.pdf")
    assert "site-8 - PRIORIDAD" in first
    assert "site-3 - PRIORIDAD" in second
    assert first != second
