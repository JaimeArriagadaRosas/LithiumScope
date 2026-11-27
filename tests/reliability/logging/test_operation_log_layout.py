"""Application operations share one logging session; no parallel legacy log roots."""
from contextlib import contextmanager
import inspect
import json

from lithiumscope.runtime import preboot_save_report
from lithiumscope.cli import scientific_lab_command, model_release_command, model_comparison_command


def test_operation_entrypoints_use_session_scoped_logging(monkeypatch, tmp_path):
    assert 'current_session_directory() / "initialization"' in inspect.getsource(preboot_save_report._save_report)
    assert 'run_log("installation","versioned_model")' in inspect.getsource(model_release_command.run)
    assert 'run_log("eda",label)' in inspect.getsource(scientific_lab_command._eda_menu)
    assert 'LITHIUMSCOPE_SESSION_DIR' in inspect.getsource(scientific_lab_command._run_tests)

    from lithiumscope.core.logging import session
    from lithiumscope.experiments.shared import registry, version_comparison, robustness_evidence

    active = []
    recorded = []

    @contextmanager
    def logged(operation, label):
        scope = (operation, label)
        assert not active
        active.append(scope)
        recorded.append(scope)
        try:
            yield
        finally:
            assert active.pop() == scope

    def saved(*, groups):
        assert active == [("simulations", "saved_version_comparison")]
        assert groups == ("model_1", "model_2")
        path = tmp_path / "saved.json"
        path.write_text(json.dumps({"state": "complete",
            "models": {group: {"state": "complete"} for group in groups}}), encoding="utf-8")
        return path

    def scenarios(*, groups, include_saved_oof):
        assert len(groups) == 1
        assert active == [("simulations", "registered_" + groups[0])]
        assert include_saved_oof is False
        path = tmp_path / (groups[0] + ".json")
        path.write_text(json.dumps({"state": "COMPLETADO"}), encoding="utf-8")
        return path

    monkeypatch.setattr(session, "run_log", logged)
    monkeypatch.setattr(version_comparison, "run_version_comparison", saved)
    monkeypatch.setattr(registry, "run_registered_simulations", scenarios)
    monkeypatch.setattr(robustness_evidence, "find_reusable_robustness", lambda _: (None, "missing"))
    monkeypatch.setattr(model_comparison_command, "choose", lambda *_: "s")

    assert model_comparison_command.run(("model_1", "model_2"))
    assert recorded == [
        ("simulations", "saved_version_comparison"),
        ("simulations", "registered_model_1"),
        ("simulations", "registered_model_2"),
    ]
    assert not active
