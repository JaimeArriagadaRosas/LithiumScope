from pathlib import Path
from types import SimpleNamespace
import json

from lithiumscope.cli import scientific_lab_command as lab
from lithiumscope.tools.inspection import health


def test_lab_menu_has_exact_dispatch_and_cancel(monkeypatch, capsys):
    calls = []
    choices = iter(["1", "2", "3", "0"])
    monkeypatch.setattr(lab, "choose", lambda *_: next(choices))
    monkeypatch.setattr(lab, "clear_screen", lambda: None)
    monkeypatch.setattr(lab, "pause", lambda: None)
    for name in ("_eda_menu", "_validation_menu", "_health"):
        monkeypatch.setattr(lab, name, lambda n=name: calls.append(n))
    lab.run()
    assert calls == ["_eda_menu", "_validation_menu", "_health"]
    out = capsys.readouterr().out
    assert "1. Análisis exploratorio" in out
    assert "2. Ejecución de tests y simulaciones" in out
    assert "3. Salud y diagnóstico del entorno" in out
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


def test_validation_menu_integral_dispatch(monkeypatch):
    calls=[]
    monkeypatch.setattr(lab,"choose",lambda *_:"4")
    for name in ("_run_tests","_run_simulations","_treatment_experiments"):
        monkeypatch.setattr(lab,name,lambda n=name:calls.append(n) or True)
    lab._validation_menu()
    assert calls==["_run_tests","_run_simulations","_treatment_experiments"]
