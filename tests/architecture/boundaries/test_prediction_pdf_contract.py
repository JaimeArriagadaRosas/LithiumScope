from pathlib import Path
import ast


def test_integrated_prediction_uses_pdf_not_html():
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()) / "src/lithiumscope"
    integrated = (root / "prediction/execution/integrated.py").read_text(encoding="utf-8")
    output = (root / "prediction/execution/outputs.py").read_text(encoding="utf-8")
    assert "prediction.execution.outputs import _write_outputs" in integrated
    assert 'session.root / "report.pdf"' in output
    assert '"report_pdf": relative(report_path)' in output
    assert "lithiumscope.reporting.service import report_title, write_pdf_report" in output
    assert "write_html_report" not in integrated + output


def test_report_service_does_not_import_reportlab_eagerly():
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()) / "src/lithiumscope/reporting"
    tree = ast.parse((root / "service.py").read_bytes())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("reportlab")
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("reportlab") for a in node.names)
