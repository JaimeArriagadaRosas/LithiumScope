import numpy as np

from lithiumscope.core.checkpoints import FoldCheckpointStore


def test_fold_checkpoint_roundtrip(tmp_path):
    store = FoldCheckpointStore(tmp_path, "random_forest")
    store.save_fold(
        1,
        [0, 2],
        [1.5, 2.5],
        {"rmse": 1.0, "mae": 0.8, "r2": 0.5},
        params={"n_estimators": 100},
    )

    payload = store.load_fold(1, [0, 2])

    assert payload is not None
    assert np.allclose(payload["predictions"], [1.5, 2.5])
    assert payload["params"]["n_estimators"] == 100


def test_fold_checkpoint_rejects_changed_split(tmp_path):
    store = FoldCheckpointStore(tmp_path, "random_forest")
    store.save_fold(
        1,
        [0, 2],
        [1.5, 2.5],
        {"rmse": 1.0},
    )

    assert store.load_fold(1, [0, 3]) is None


def test_checkpoint_detects_modified_payload(tmp_path):
    import json
    store = FoldCheckpointStore(tmp_path, "rf")
    path = store.save_fold(1, [0, 2], [1.5, 2.5], {"rmse": 1.0})
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["predictions"][0] = 100
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert store.load_fold(1, [0, 2]) is None


def test_checkpoint_rejects_nonfinite_and_duplicate_indices(tmp_path):
    import pytest
    store = FoldCheckpointStore(tmp_path, "rf")
    with pytest.raises(ValueError):
        store.save_fold(1, [0, 2], [1.5, np.nan], {})
    with pytest.raises(ValueError):
        store.save_fold(1, [0, 0], [1.5, 2.5], {})


def test_algorithm_checkpoint_detects_modification(tmp_path):
    import json
    store = FoldCheckpointStore(tmp_path, "rf")
    path = store.save_algorithm({"params": {"depth": 3}})
    assert store.load_algorithm()["params"]["depth"] == 3
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload"]["params"]["depth"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    assert store.load_algorithm() is None



def test_bound_checkpoint_identity_prevents_cross_run_reuse(tmp_path):
    import shutil
    import pytest
    store=FoldCheckpointStore(tmp_path,"rf",identity="source_a")
    fold=store.save_fold(1,[0,2],[1.,2.],{"rmse":1.})
    (store.root/"optuna_final.db").write_bytes(b"study")
    with pytest.raises(ValueError,match="different evaluation identity"):
        FoldCheckpointStore(tmp_path,"rf",identity="source_b")
    other=FoldCheckpointStore(tmp_path,"other",identity="source_b")
    shutil.copyfile(fold,other.fold_path(1))
    assert other.load_fold(1,[0,2]) is None
    assert FoldCheckpointStore(tmp_path,"rf",identity="source_a").load_fold(1,[0,2])
    with pytest.raises(ValueError,match="integers"):
        store.save_fold(2,[1.5],[2.],{})
    with pytest.raises(ValueError,match="integers"):
        store.load_fold(1,[0.,2.])


def test_historical_checkpoint_readonly_not_resumable(tmp_path):
    import json
    import pytest
    from lithiumscope.core.checkpoints import read_checkpoint_evidence
    old=FoldCheckpointStore(tmp_path,"legacy")
    path=old.fold_path(1)
    payload={"fold":1,"test_indices":[0,2],"predictions":[1.,2.],"metrics":{"rmse":1.}}
    path.write_text(json.dumps(payload),encoding="utf-8")
    original=path.read_bytes()
    assert read_checkpoint_evidence(path,allow_legacy=True)==payload
    with pytest.raises(ValueError,match="Unsealed"):
        read_checkpoint_evidence(path)
    assert old.load_fold(1,[0,2]) is None
    with pytest.raises(ValueError,match="historical evidence"):
        FoldCheckpointStore(tmp_path,"legacy",identity="new")
    assert path.read_bytes()==original
    sealed=old.save_fold(2,[1,3],[1.,2.],{})
    assert read_checkpoint_evidence(sealed)["fold"]==2
    data=json.loads(sealed.read_text(encoding="utf-8"))
    data["payload"]["fold"]=99
    sealed.write_text(json.dumps(data),encoding="utf-8")
    with pytest.raises(ValueError,match="Invalid checkpoint evidence seal"):
        read_checkpoint_evidence(sealed,allow_legacy=True)


def test_evaluation_identity_changes_with_order_targets_policy_and_groups(tmp_path):
    import pandas as pd
    from lithiumscope.core.scientific.checkpoint_identity import evaluation_identity
    x=pd.DataFrame({"a":[1.,2.,3.,4.],"cat":["a","b","a","b"]})
    y=pd.Series([1.,2.,3.,4.])
    groups=pd.Series(["a","a","b","b"])
    splits=[([0,1],[2,3]),([2,3],[0,1])]
    root=tmp_path/"checkpoints"
    kwargs=dict(x=x,y=y,groups=groups,splits=splits,policy={"seed":42},extra={"qa":[True]*4})
    original=evaluation_identity(root,**kwargs)
    assert original==evaluation_identity(root,**kwargs)
    for key,value in [("y",y+1),("x",x.iloc[::-1]),("groups",groups.str.upper()),
                      ("policy",{"seed":43}),("splits",splits[::-1]),("extra",{"qa":[True,False,True,True]})]:
        assert evaluation_identity(root,**{**kwargs,key:value})!=original
