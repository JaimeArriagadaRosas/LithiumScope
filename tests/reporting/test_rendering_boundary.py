from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from lithiumscope.prediction.analysis.decision import calculate_decision
from lithiumscope.prediction.exports.decision_previews import prepare_decision_previews
from lithiumscope.reporting.sections.decision import build_decision_page
from lithiumscope.reporting.styles import build_styles


def test_renderer_handles_insufficient_columns_without_fake_rank():
    decision = calculate_decision(pd.DataFrame({"case_id": ["a", "b", "c"]}))
    story = build_decision_page(SimpleNamespace(decision=decision), build_styles())
    assert any(decision.reason in p.getPlainText() for p in story if hasattr(p, "getPlainText"))


def test_previews_require_documented_band_order(tmp_path):
    selected = pd.DataFrame({"case_id": ["x"], "image": [str(tmp_path / "patch.npy")]})
    assert prepare_decision_previews(selected, {}, tmp_path / "rgb") == {}
    assert not (tmp_path / "rgb").exists()


def test_previews_use_explicit_band_order(tmp_path, monkeypatch):
    import numpy as np
    from lithiumscope.prediction.exports import charts
    patch = tmp_path / "patch.npy"
    patch.touch()
    observed = []
    def preview(path, bands):
        observed.append((path, bands))
        return np.ones((4, 4, 3), dtype=float)
    monkeypatch.setattr(charts, "_rgb_preview", preview)
    bands = ["blue", "green", "red"]
    selected = pd.DataFrame({"case_id": ["../case"], "image": [str(patch)]})
    results = prepare_decision_previews(selected, {"bands": bands}, tmp_path / "rgb")
    assert observed == [(patch, bands)]
    assert results["../case"].parent == tmp_path / "rgb"
    assert results["../case"].read_bytes().startswith(b"\x89PNG")


def test_renderer_does_not_import_raw_patch_extractors():
    source = Path(__file__).resolve().parents[2] / "src/lithiumscope/reporting/sections/decision.py"
    assert "load_imagery" not in source.read_text()
    assert "_rgb_preview" not in source.read_text()
