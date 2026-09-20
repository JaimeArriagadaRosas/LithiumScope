from pathlib import Path


def test_integrated_prediction_uses_pdf_not_html():
    root = Path(__file__).resolve().parents[1]
    integrated = (
        root
        / "src"
        / "lithiumscope"
        / "prediction"
        / "integrated.py"
    ).read_text(encoding="utf-8")
    report = (
        root
        / "src"
        / "lithiumscope"
        / "prediction"
        / "report.py"
    ).read_text(encoding="utf-8")

    assert 'session.root / "report.pdf"' in integrated
    assert '"report_pdf": str(report_path)' in integrated
    assert "def write_pdf_report" in report
    assert "write_html_report" not in integrated


def test_report_facade_does_not_import_reportlab_eagerly():
    root = Path(__file__).resolve().parents[1]
    report = (
        root
        / "src"
        / "lithiumscope"
        / "prediction"
        / "report.py"
    ).read_text(encoding="utf-8")

    assert "from reportlab" not in report
    assert "import reportlab" not in report
