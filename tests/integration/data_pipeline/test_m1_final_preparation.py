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


def test_recovery_identity_does_not_read_lithium_and_blocks_ambiguity():
    import pandas as pd
    from lithiumscope.experiments.data_lab.preparation.recovery import independent_links
    row = {"source_sample":"A", "Latitude":-20., "Longitude":-70.,
           "SiO2":50., "TiO2":1., "Al2O3":15., "MnO":.1, "MgO":5.}
    base = pd.DataFrame([{**row,"Li_icpms":10.}])
    new = pd.DataFrame([{**row,"Li_icpms":999.}])
    found = independent_links(base,new)
    assert len(found)==1 and found.unique_link.all() and not found.target_used.any()
    ambiguous = independent_links(base,pd.concat([new,new],ignore_index=True))
    assert len(ambiguous)==2 and not ambiguous.unique_link.any()
