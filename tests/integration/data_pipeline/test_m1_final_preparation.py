from pathlib import Path

from lithiumscope.experiments.data_lab.preparation import dispositions


def test_source_decisions_fail_closed_and_do_not_re_retire_reopened(monkeypatch, tmp_path):
    key = "chen_global_arcs_2020"
    history = tmp_path / "history" / key
    history.mkdir(parents=True)
    (history / "reopen.json").write_text("{}")
    monkeypatch.setattr(dispositions.source_catalog, "retirement", lambda *a, **kw: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("No retirement without verified complete evidence")
    monkeypatch.setattr(dispositions.source_catalog, "retire", forbidden)
    profiles = {key: {"path": Path("profile"), "record": {"manifest_sha256": "hash"},
                      "manifest": {"source_sha256": {"file": "sha"}, "summary": {}}, "history": []}}
    rows = dispositions.apply_decisions(profiles, {}, tmp_path)
    assert next(r for r in rows if r["source"] == key)["state"] == "reopened_requires_review"
    assert all(r["state"] == "blocked" for r in rows if r["source"] != key)
    assert all(r["approved_rows"] == 0 for r in rows)
