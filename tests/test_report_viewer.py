from pathlib import Path

import lithiumscope.cli.report_viewer as viewer


def test_report_opens_in_default_browser(monkeypatch, tmp_path: Path):
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF")

    opened = {}

    def fake_open(uri):
        opened["uri"] = uri
        return True

    monkeypatch.setattr(viewer.webbrowser, "open_new_tab", fake_open)

    assert viewer.open_report_in_default_browser(report) is True
    assert opened["uri"].startswith("file:")
