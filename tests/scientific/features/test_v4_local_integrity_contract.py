from types import SimpleNamespace
import pandas as pd
import pytest
from lithiumscope.model_2.preparation.features import v4_integrity
from lithiumscope.tools.acquisition.sentinel2 import rededge


def test_missing_cache_never_downloads(monkeypatch,tmp_path):
    image=tmp_path/"base.npy"
    image.write_bytes(b"base")
    sample=SimpleNamespace(sample_id="a",sentinel_scene_id="scene",image_path=str(image),latitude=-20,longitude=-70)
    monkeypatch.setattr(rededge,"scene_item",lambda *a,**k: pytest.fail("No scene lookup for absent patch"))
    with pytest.raises(FileNotFoundError,match="red-edge"):
        rededge.cached_rededge_sample(sample,output=tmp_path/"cache",allow_acquisition=False)


def test_derived_values_and_qa_must_match(monkeypatch,tmp_path):
    manifest=tmp_path/"manifest.csv"; matrix=tmp_path/"matrix.csv"; quality=tmp_path/"quality.csv"
    pd.DataFrame({"sample_id":["a"]}).to_csv(manifest,index=False)
    features={f"f{i}":float(i) for i in range(44)}
    pd.DataFrame([features]).to_csv(matrix,index=False)
    qa={"sample_id":"a","scene_id":"s","rededge_patch_sha256":"red","original_patch_sha256":"base",
        "scene_band_hash":"scene","rededge_joint_valid_fraction":1.}
    pd.DataFrame([qa]).to_csv(quality,index=False)
    def cached(sample,**kwargs):
        assert kwargs["allow_acquisition"] is False
        return dict(features,f0=999),qa
    monkeypatch.setattr(v4_integrity,"cached_rededge_sample",cached)
    with pytest.raises(ValueError,match="values differ"):
        v4_integrity.verify_rededge_cohort(manifest,matrix,quality)
