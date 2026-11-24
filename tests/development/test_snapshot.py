import json

from lithiumscope.tools.inspection.development import inventory, snapshot


def test_inventory_never_reads_outside_workspace(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    secret = tmp_path / "private.txt"
    secret.write_text("do-not-export")
    assert inventory.file_record(root, secret) == {
        "state": "BLOCKED", "reason": "outside_workspace_or_invalid_path"}
    assert inventory.file_record(root, "../private.txt")["state"] == "BLOCKED"
    record = inventory.file_record(root, "missing.txt")
    assert record == {"state": "MISSING", "path": "missing.txt"}
    (root / "data.txt").write_text("local sample")
    record = inventory.file_record(root, "data.txt")
    assert record["state"] == "OK" and len(record["sha256"]) == 64
    assert "local sample" not in json.dumps(record)


def test_test_summary_distinguishes_unknown_commit_and_coverage(tmp_path):
    folder = tmp_path / "results/tests/run"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({
        "state": "passed", "counts": {"tests": 4, "failures": 0, "errors": 0, "skipped": 1},
        "log": "logs/test.log"}))
    (folder / "coverage.xml").write_text('<coverage line-rate="0.57"/>')
    report = inventory.latest_test_report(tmp_path, "a"*40)
    assert report["state"] == "OK"
    assert report["counts"]["tests"] == 4
    assert report["matches_current_clean_commit"] is None
    assert report["line_coverage_pct"] == 57
    assert report["log"]["state"] == "MISSING"


def test_section_preserves_failure_without_exporting_private_details(capsys):
    def broken():
        raise ValueError("secret private path")
    report = snapshot._section("check", broken)
    assert report["state"] == "ERROR" and report["error_type"] == "ValueError"
    assert "secret private path" not in json.dumps(report)
    assert "secret private path" not in capsys.readouterr().out


def test_write_snapshot_is_unique_and_only_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "collect_snapshot", lambda root: {"state": "PARTIAL"})
    first = snapshot.write_snapshot(tmp_path)
    second = snapshot.write_snapshot(tmp_path)
    assert first != second
    assert json.loads(first.read_text()) == {"state": "PARTIAL"}
    assert first.is_relative_to(tmp_path / "results/diagnostics")
    assert len(list(tmp_path.rglob("*.json"))) == 2


def test_temporary_option_dispatch(monkeypatch, capsys):
    from lithiumscope.cli import scientific_lab_command as lab
    choices = iter(["4", "0"])
    calls = []
    monkeypatch.setattr(lab, "choose", lambda *_: next(choices))
    monkeypatch.setattr(lab, "clear_screen", lambda: None)
    monkeypatch.setattr(lab, "pause", lambda: None)
    monkeypatch.setattr(lab, "_development_snapshot", lambda: calls.append("snapshot"))
    lab.run()
    assert calls == ["snapshot"]
    assert "4. Evidencias de desarrollo (temporal)" in capsys.readouterr().out
