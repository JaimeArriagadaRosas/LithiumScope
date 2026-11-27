from pathlib import Path
from types import SimpleNamespace
import json

from lithiumscope.cli import scientific_lab_command as lab
from lithiumscope.tools.inspection import health


def test_lab_menu_has_exact_dispatch_and_cancel(monkeypatch, capsys):
    calls = []
    choices = iter(["1", "2", "3", "4", "0"])
    monkeypatch.setattr(lab, "choose", lambda *_: next(choices))
    monkeypatch.setattr(lab, "clear_screen", lambda: None)
    monkeypatch.setattr(lab, "pause", lambda: None)
    for name in ("_eda_menu", "_preparation_menu", "_validation_menu", "_health"):
        monkeypatch.setattr(lab, name, lambda n=name: calls.append(n))
    lab.run()
    assert calls == ["_eda_menu", "_preparation_menu", "_validation_menu", "_health"]
    out = capsys.readouterr().out
    assert "1. Análisis exploratorio" in out
    assert "2. Preparación y tratamiento de datos" in out
    assert "3. Ejecución de tests y simulaciones" in out
    assert "4. Salud y diagnóstico del entorno" in out
    assert "4. Evidencias de desarrollo" not in out
    assert "5. Optimización científica" not in out


def test_eda_both_preserves_partial_results(monkeypatch, capsys, tmp_path):
    from lithiumscope.cli import eda_verification
    monkeypatch.setattr(eda_verification,"RESULTS_DIR",tmp_path)
    calls = []
    monkeypatch.setattr(lab, "choose", lambda *_: "3")
    def first():
        calls.append("m1")
        raise FileNotFoundError("missing input")
    monkeypatch.setattr(lab, "_run_eda_m1", first)
    monkeypatch.setattr(lab, "_run_eda_m2", lambda: calls.append("m2") or Path("output"))
    lab._eda_menu()
    assert calls == ["m1", "m2"]
    out = capsys.readouterr().out
    assert "BLOQUEADO" in out and "PARCIAL" in out


def test_health_audit_does_not_modify_workspace(tmp_path, monkeypatch):
    source = tmp_path / "workspace"
    (source / "src").mkdir(parents=True)
    (source / "src" / "big.py").write_text("x = 1\n" * 301)
    original = (source / "src" / "big.py").read_bytes()
    monkeypatch.setattr(health, "PROJECT_ROOT", source)
    monkeypatch.setattr(health, "CONFIG_DIR", source / "config")
    monkeypatch.setattr(health, "DATA_DIR", source / "data")
    monkeypatch.setattr(health, "MODELS_DIR", source / "models")
    monkeypatch.setattr(health, "LOGS_DIR", source / "logs")
    monkeypatch.setattr(health, "RESULTS_DIR", source / "results")
    output = tmp_path / "diagnostics"
    report = health.write_health_report(output)
    record = json.loads(report.read_text(encoding="utf-8"))
    assert record["mode"] == "read_only"
    assert record["architecture"]["files_over_300"][0]["path"] == "src/big.py"
    assert (source / "src" / "big.py").read_bytes() == original
    assert sorted(p.name for p in source.iterdir()) == ["src"]


def test_validation_menu_exact_dispatch_and_cancel(monkeypatch, capsys):
    calls = []
    options = iter(["1", "2", "3", "0"])
    def select(prompt, allowed):
        assert allowed == set("0123")
        return next(options)
    monkeypatch.setattr(lab, "choose", select)
    monkeypatch.setattr(lab, "clear_screen", lambda: None)
    for name in ("_run_tests", "_comparison_menu", "_treatment_menu"):
        monkeypatch.setattr(lab, name, lambda n=name: calls.append(n) or True)
    for _ in range(4):
        lab._validation_menu()
    assert calls == ["_run_tests", "_comparison_menu", "_treatment_menu"]
    out = capsys.readouterr().out
    assert "1. Validación del software\n2. Comparaciones de modelos\n3. Evaluación de tratamiento de datos\n0. Volver" in out
    assert "Ejecutar todo" not in out and "OASIS" not in out


def test_comparison_menu_selects_only_requested_model(monkeypatch):
    from lithiumscope.cli import model_comparison_command
    monkeypatch.setattr(lab, "choose", lambda *_: "2")
    monkeypatch.setattr(lab, "clear_screen", lambda: None)
    calls = []
    monkeypatch.setattr(model_comparison_command, "run", lambda groups: calls.append(groups) or True)
    assert lab._comparison_menu() is True
    assert calls == [("model_2",)]


def test_treatment_menu_m1_dispatch(monkeypatch):
    from lithiumscope.cli import m1_targeted_command
    choices = iter(["1", "f"])
    monkeypatch.setattr(lab, "choose", lambda *_: next(choices))
    calls = []
    monkeypatch.setattr(lab, "clear_screen", lambda: calls.append("clear"))
    monkeypatch.setattr(m1_targeted_command, "run", lambda: calls.append("m1") or True)
    assert lab._treatment_menu() is True
    assert calls == ["clear", "m1"]


def test_treatment_menu_both_selects_both(monkeypatch):
    monkeypatch.setattr(lab, "choose", lambda *_: "3")
    monkeypatch.setattr(lab, "clear_screen", lambda: None)
    calls = []
    monkeypatch.setattr(lab, "_treatment_experiments", lambda **kw: calls.append(kw) or True)
    assert lab._treatment_menu() is True
    assert calls == [{"groups": ("model_1", "model_2")}]


def test_validation_clears_screen_before_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(lab, "choose", lambda *_: "1")
    monkeypatch.setattr(lab, "clear_screen", lambda: calls.append("clear"))
    monkeypatch.setattr(lab, "_run_tests", lambda: calls.append("tests") or True)
    lab._validation_menu()
    assert calls == ["clear", "tests"]


def test_targeted_m1_runs_approved_reference_without_extra_prompts(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from lithiumscope.cli import m1_targeted_command as command
    from lithiumscope.experiments.model_1.protocols import evidence as protocol
    from lithiumscope.experiments.model_1.studies.balanced import runner
    from lithiumscope.core.logging import session

    calls = []
    reference = "training_20261125_115932_m0300_826123c1"
    monkeypatch.setattr(protocol, "available_runs", lambda: ["newer-unrelated", reference, "old"])
    monkeypatch.setattr(session, "run_log", lambda *args: nullcontext())
    def forbidden_input(*args):
        raise AssertionError("No additional input expected")
    monkeypatch.setattr("builtins.input", forbidden_input)
    (tmp_path/"manifest.json").write_text('{"state":"complete"}', encoding="utf-8")
    monkeypatch.setattr(runner, "run_balanced", lambda run_id: calls.append(run_id) or tmp_path)
    assert command.run() is True
    assert calls == [reference]
    monkeypatch.setattr(protocol, "available_runs", lambda: ["old"])
    assert command.run() is False
    assert calls == [reference]


def test_software_menu_executes_full_preflight_runner(monkeypatch, tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/run_tests.py").write_text("pass")
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr(lab, "PROJECT_ROOT", tmp_path)
    from lithiumscope.core.logging import session
    monkeypatch.setattr(session, "current_session_directory", lambda: tmp_path / "logs/sessions/test")
    captured = {}
    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(lab.subprocess, "run", run)
    assert lab._run_tests() is False
    assert "--preflight" in captured["command"] and "--coverage" in captured["command"]
    assert "--continue-on-collection-errors" in captured["command"]
    assert captured["cwd"] == tmp_path
