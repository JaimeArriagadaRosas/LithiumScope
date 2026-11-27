from reportlab.pdfgen.canvas import Canvas

from lithiumscope.tools.inspection.pdf_integrity import inspect_pdf
from lithiumscope.results.evidence import latest_document
from lithiumscope.reporting.formatting import number, fraction, nonnegative_interval, operating_classification


def test_readonly_structural_check(tmp_path):
    path = tmp_path / "report.pdf"
    canvas = Canvas(str(path))
    canvas.drawString(10, 20, "synthetic")
    canvas.save()
    original = path.read_bytes()
    result = inspect_pdf(path)
    assert result["state"] in {"SCREENED", "VERIFIED_STRUCTURE"}
    if result["state"] == "VERIFIED_STRUCTURE":
        assert result["page_count"] == 1
    assert not result["visual_review"]
    assert path.read_bytes() == original


def test_fake_pdf_trailer_does_not_count_as_evidence(tmp_path):
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"%PDF-1.4\nstartxref\n99999\n%%EOF\n")
    assert inspect_pdf(path)["state"] == "INVALID"
    assert latest_document(tmp_path, ".pdf") is None


def test_undefined_or_invalid_display_values_are_not_predictions():
    assert number(float("nan")) == "N/D"
    assert number(float("inf")) == "N/D"
    assert fraction(1.1) == "N/D"
    assert nonnegative_interval(5, 2) == "N/D"
    assert operating_classification(1.2, .5) == "N/D"
    assert operating_classification(.175, .088) == "POSITIVA"
