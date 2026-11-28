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


def test_ilr_is_orthonormal_scale_invariant_and_train_only():
    import numpy as np
    import pandas as pd
    from lithiumscope.experiments.data_lab.preparation.variants import fit_variant, transform, helmert
    from lithiumscope.model_1.training.preprocessing.tabular import OXIDES
    train=pd.DataFrame([np.arange(1,11),np.arange(2,12)],columns=OXIDES,dtype=float)
    reserved=train.iloc[:1].copy()*10
    state=fit_variant(train,OXIDES,[],"ilr_oxides")
    assert np.allclose(helmert(10).T@helmert(10),np.eye(9))
    assert np.allclose(transform(train.iloc[:1],state),transform(reserved,state))
    reserved.iloc[0,0]=0
    reserved.iloc[0,1]=np.nan
    assert np.isfinite(transform(reserved,state).to_numpy()).all()
    assert state["zero_fill"][OXIDES[0]]==.5
    assert state["missing_fill"][OXIDES[1]]==2.5


def test_missing_indicators_use_only_train_and_categories_are_ablated():
    import pandas as pd
    from lithiumscope.experiments.data_lab.preparation.variants import fit_variant,transform
    train=pd.DataFrame({"SiO2":[1.,2.],"MgO":[None,1.],"Rock_type":["a","b"]})
    held=pd.DataFrame({"SiO2":[None],"MgO":[None],"Rock_type":["c"]})
    state=fit_variant(train,["SiO2","MgO"],["Rock_type"],"chemistry_missing_indicators")
    assert state["indicator_columns"]==["MgO"]
    assert transform(held,state).missing_MgO.tolist()==[1.]
    assert "missing_SiO2" not in transform(held,state)
    ablation=fit_variant(train,["SiO2","MgO"],["Rock_type"],"without_categories")
    assert list(transform(held,ablation))==["SiO2","MgO"]
