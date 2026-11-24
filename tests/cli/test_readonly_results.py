from reportlab.pdfgen.canvas import Canvas
from openpyxl import Workbook

from lithiumscope.results import evidence
from lithiumscope.cli import results_command


def test_latest_evidence_rejects_invalid_and_external_files(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    canvas = Canvas(str(old / "report.pdf"))
    canvas.drawString(20, 20, "test")
    canvas.save()
    (new / "report.pdf").write_bytes(b"corrupt")
    assert evidence.latest_document(tmp_path, ".pdf") == old / "report.pdf"
    Workbook().save(old / "book.xlsx")
    (new / "book.xlsx").write_bytes(b"PKinvalid")
    assert evidence.latest_document(tmp_path, ".xlsx") == old / "book.xlsx"


def test_results_menu_only_routes_read_operations(monkeypatch, capsys):
    calls = []
    for name in ("_pdf", "_excel", "_models", "_eda", "_tests"):
        monkeypatch.setattr(results_command, name, lambda n=name: calls.append(n))
    for option in ("1", "2", "3", "4", "5", "0"):
        monkeypatch.setattr(results_command, "choose", lambda *_, value=option: value)
        results_command.run()
    assert calls == ["_pdf", "_excel", "_models", "_eda", "_tests"]
    assert "promover" not in capsys.readouterr().out.lower()


def test_active_model_query_does_not_create_files(tmp_path, monkeypatch):
    from lithiumscope.persistence import active_models
    monkeypatch.setattr(active_models, "ACTIVE_MODELS_PATH", tmp_path / "models" / "active_models.json")
    assert evidence.active_summary("model_1")["state"] == "N/D"
    assert list(tmp_path.iterdir()) == []


def test_latest_test_summary_ignores_invalid_json(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "summary.json").write_text('{"run_id": "run", "counts": {"tests": 2}}')
    broken = tmp_path / "new"
    broken.mkdir()
    (broken / "summary.json").write_text("{")
    assert evidence.latest_record(tmp_path, "summary.json")[1]["run_id"] == "run"
