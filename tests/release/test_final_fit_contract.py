import json
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor, DummyClassifier

from lithiumscope.core.exceptions import ModelNotReadyError
from lithiumscope.persistence.final_fit_contract import freeze_final_fit
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.persistence import save_model, load_model


def fixture_data():
    x=pd.DataFrame({"x":[1.,2.,3.,4.]},index=[10,12,15,20])
    y=pd.Series([0.,1.,0.,1.],index=x.index)
    g=pd.Series(["a","b","a","b"],index=x.index)
    kw={"x":x,"y":y,"groups":g,"row_ids":["id1","id2","id3","id4"],
        "dataset_sha256":"a"*64,"training_signature":"b"*64,"params":{},"policy":{"algorithm":"dummy"}}
    return x,y,kw


def test_final_fit_excludes_reserved_and_binds_input():
    x,y,kw=fixture_data()
    first=freeze_final_fit(**kw,reserved_ids=["external"])
    assert first["manifest"]["external_holdout_reserved"]
    assert first!=freeze_final_fit(**{**kw,"y":y+1},reserved_ids=["external"])
    with pytest.raises(ValueError,match="intersects"):
        freeze_final_fit(**kw,reserved_ids=["id2"])
    with pytest.raises(ValueError,match="target is not aligned"):
        freeze_final_fit(**{**kw,"y":y.reset_index(drop=True)})


@pytest.mark.parametrize("group",["model_1","model_2"])
def test_atomic_candidate_roundtrip_and_no_implicit_activation(monkeypatch,tmp_path,group):
    monkeypatch.setattr(save_model,"MODELS_DIR",tmp_path/"models")
    monkeypatch.setattr(load_model,"active_model_path",lambda _:None)
    x,y,kw=fixture_data()
    fit=freeze_final_fit(**kw)
    est=DummyRegressor().fit(x,y) if group=="model_1" else DummyClassifier(strategy="prior").fit(x,y.astype(int))
    bundle={"estimator":est,"experimental_only":True,"final_fit_contract":fit}
    if group=="model_1":
        bundle["schema"]=FeatureSchema(["x"],[],2)
    else:
        bundle["features"]=["x"]
    meta={"final_fit_contract_sha256":fit["identity_sha256"],"experimental_only":True}
    path,metadata=save_model.save_model_bundle(group,"candidate",bundle,meta,run_id="new",verification_probe=x)
    assert json.loads(metadata.read_text())["serialization_verification"]["maximum_absolute_delta"]==0
    assert (path.parent/"final_fit_contract.json").exists()
    assert (path.parent/"artifact_integrity.json").exists()
    original=path.read_bytes()
    with pytest.raises(FileExistsError,match="overwrite"):
        save_model.save_model_bundle(group,"candidate",bundle,meta,run_id="new",verification_probe=x)
    assert path.read_bytes()==original
    with pytest.raises(ModelNotReadyError,match="explicitly active"):
        load_model.latest_model_path(group)
    with pytest.raises(ValueError,match="cannot be activated"):
        save_model.save_model_bundle(group,"bad",bundle,meta,run_id="bad",activate=True)
    assert not list(path.parent.parent.glob(".staging_*"))


def test_bad_serialization_does_not_publish(monkeypatch,tmp_path):
    monkeypatch.setattr(save_model,"MODELS_DIR",tmp_path/"models")
    x,y,kw=fixture_data()
    fit=freeze_final_fit(**kw)
    bundle={"estimator":DummyRegressor().fit(x,y),"schema":FeatureSchema(["x"],[],2),
            "experimental_only":True,"final_fit_contract":fit}
    monkeypatch.setattr(save_model,"verify_serialized_candidate",lambda *a,**k:(_ for _ in ()).throw(ValueError("parity failed")))
    with pytest.raises(ValueError,match="parity failed"):
        save_model.save_model_bundle("model_1","bad",bundle,{"final_fit_contract_sha256":fit["identity_sha256"]},
                                    run_id="bad",verification_probe=x)
    root=tmp_path/"models/model_1/trained"
    assert not (root/"bad").exists() and not list(root.glob(".staging_*"))
