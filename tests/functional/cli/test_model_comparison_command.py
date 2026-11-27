"""Saved evidence first; complementary scenarios require explicit approval."""
from contextlib import nullcontext
import json

import pytest

from lithiumscope.cli import model_comparison_command as command
from lithiumscope.core.logging import session
from lithiumscope.experiments.shared import registry, version_comparison


@pytest.mark.parametrize("answer", ["n", "s"])
def test_comparison_uses_selected_model_and_never_repeats_saved_oof(monkeypatch, tmp_path, answer):
    calls = []
    monkeypatch.setattr(session, "run_log", lambda *args: nullcontext())
    monkeypatch.setattr(command, "choose", lambda *_: answer)
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
