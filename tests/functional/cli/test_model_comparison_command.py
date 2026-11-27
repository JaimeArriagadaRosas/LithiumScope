"""Saved evidence first; complementary scenarios require explicit approval."""
from contextlib import nullcontext
import json

import pytest

from lithiumscope.cli import model_comparison_command as command
from lithiumscope.core.logging import session
from lithiumscope.experiments.shared import registry, version_comparison, robustness_evidence


@pytest.mark.parametrize("answer", ["n", "s"])
def test_comparison_uses_selected_model_and_never_repeats_saved_oof(monkeypatch, tmp_path, answer):
    calls = []
    monkeypatch.setattr(session, "run_log", lambda *args: nullcontext())
    monkeypatch.setattr(command, "choose", lambda *_: answer)
    monkeypatch.setattr(robustness_evidence, "find_reusable_robustness", lambda _: (None, "missing"))
    def saved(**kwargs):
        calls.append(("saved", kwargs))
        path = tmp_path / "saved.json"
        path.write_text(json.dumps({"state": "complete", "models": {"model_2": {"state": "complete"}}}))
        return path
    def scenarios(**kwargs):
        calls.append(("scenarios", kwargs))
        path = tmp_path / "scenarios.json"
        path.write_text(json.dumps({"state": "COMPLETADO"}))
        return path
    monkeypatch.setattr(version_comparison, "run_version_comparison", saved)
    monkeypatch.setattr(registry, "run_registered_simulations", scenarios)
    assert command.run(("model_2",))
    assert calls[0] == ("saved", {"groups": ("model_2",)})
    assert calls[1:] == ([("scenarios", {"groups": ("model_2",), "include_saved_oof": False})]
                         if answer == "s" else [])


def test_saved_comparison_scope_never_discovers_other_model(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(version_comparison, "discover_models", lambda group: (calls.append(group) or {}, []))
    def missing(*args):
        raise FileNotFoundError("OOF unavailable")
    monkeypatch.setattr(version_comparison, "compare_saved", missing)
    manifest = version_comparison.run_version_comparison(output_root=tmp_path, groups=("model_1",))
    result = json.loads(manifest.read_text())
    assert calls == ["model_1"]
    assert list(result["models"]) == ["model_1"]
    assert result["state"] == "partial"

def test_reuses_verified_scenarios_without_prompt_or_rerun(monkeypatch, tmp_path):
    monkeypatch.setattr(session, "run_log", lambda *args: nullcontext())
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps({"state": "complete", "models": {"model_1": {"state": "complete"}}}))
    monkeypatch.setattr(version_comparison, "run_version_comparison", lambda **kw: saved)
    monkeypatch.setattr(robustness_evidence, "find_reusable_robustness", lambda _: (tmp_path / "cached.json", None))
    def forbidden(*args, **kwargs):
        pytest.fail("No repeated scenario run or prompt expected")
    monkeypatch.setattr(command, "choose", forbidden)
    monkeypatch.setattr(registry, "run_registered_simulations", forbidden)
    assert command.run(("model_1",))


def test_reuse_requires_identity_and_unchanged_output_bytes(monkeypatch, tmp_path):
    from lithiumscope.experiments.shared import comparison_models
    folder = tmp_path / "sim_example"
    folder.mkdir()
    output = folder / "metrics.csv"
    output.write_text("score\n1\n")
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps({"state": "COMPLETADO", "groups": ["model_1"],
        "include_saved_oof": False, "input_identity": "match",
        "output_hashes": robustness_evidence.output_hashes(folder)}))
    monkeypatch.setattr(comparison_models, "discover_models", lambda _: ({}, []))
    monkeypatch.setattr(robustness_evidence, "input_identity", lambda *args: ("match", None))
    assert robustness_evidence.find_reusable_robustness("model_1", root=tmp_path)[0] == manifest
    output.write_text("score\n2\n")
    assert robustness_evidence.find_reusable_robustness("model_1", root=tmp_path)[0] is None
