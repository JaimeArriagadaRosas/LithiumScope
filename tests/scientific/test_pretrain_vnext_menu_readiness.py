"""Training readiness without executing the actual M1/M2 competitions."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from lithiumscope.datasets.georoc_ingestion import ingest_georoc_files
from lithiumscope.datasets import provisioner
from lithiumscope.core.device import DeviceInfo
from lithiumscope.model_2.training.factory import create_model, get_algorithm


def _georoc_rows():
    rows=[]
    for index, feot in enumerate((7.0, np.nan),start=1):
        rows.append({
            "SAMPLE NAME":f"S-{index}",
            "TYPE OF MATERIAL":"WHOLE ROCK",
            "LONGITUDE MIN":-70.25+index*.05,
            "LATITUDE MIN":-23.24+index*.05,
            "LI(PPM)":18+index,
            "SIO2(WT%)":58., "TIO2(WT%)":1.,
            "AL2O3(WT%)":17., "FE2O3(WT%)":4.,
            "FEOT(WT%)":feot, "MNO(WT%)":.2,
            "MGO(WT%)":4., "CAO(WT%)":7.,
            "NA2O(WT%)":4., "K2O(WT%)":3.,
            "P2O5(WT%)":.6,
        })
    return pd.DataFrame(rows)


def test_georoc_b_prefers_total_feot_and_never_adds_iron_twice(tmp_path: Path):
    raw=tmp_path/"raw.csv"
    _georoc_rows().to_csv(raw,index=False)
    expanded_path=tmp_path/"B.csv"
    a_path=tmp_path/"A.csv"
    b=ingest_georoc_files(
        [raw],expanded_path,iron_total_mode="feot_equivalent"
    )
    ingest_georoc_files([raw],a_path,iron_total_mode="original")
    result=pd.read_csv(b.path)
    original=pd.read_csv(a_path)
    assert len(result)==2
    assert result["Fe2O3"].iloc[0]==pytest.approx(7*1.111348)
    assert original["Fe2O3"].iloc[0]==pytest.approx(4)
    assert result["Fe2O3"].iloc[1]==pytest.approx(4)
    assert result["Fe_total_provenance"].tolist()==[
        "FEOT_CONVERTED_TO_FE2O3T","ORIGINAL_FE2O3"
    ]
    assert not (result["Fe2O3"].iloc[0]==pytest.approx(11.778))
    assert result["SUM (no water)"].iloc[0]==pytest.approx(
        result.loc[0,["SiO2","TiO2","Al2O3","Fe2O3","MnO",
                      "MgO","CaO","Na2O","K2O","P2O5"]].sum()
    )
    with pytest.raises(ValueError):
        ingest_georoc_files([raw],tmp_path/"bad.csv",iron_total_mode="guess")


def test_frozen_m2_manifest_is_read_only_and_checks_integrity(tmp_path,monkeypatch):
    manifest=tmp_path/"manifest.csv"
    pd.DataFrame({
        "Li_icpms":[9.,22.,30.],
        "spatial_group":["a","b","c"],
        "image_path":["one.npy","two.npy","three.npy"],
    }).to_csv(manifest,index=False)
    expected=hashlib.sha256(manifest.read_bytes()).hexdigest()
    cfg={"training":{
        "manifest_path":str(manifest),
        "frozen_expected_rows":3,
        "frozen_manifest_sha256":expected,
    }}
    monkeypatch.setattr(provisioner,"load_config",lambda name:cfg)
    original=manifest.read_bytes()
    assert provisioner.require_frozen_model_2_manifest()==manifest
    assert manifest.read_bytes()==original
    cfg["training"]["frozen_expected_rows"]=2
    with pytest.raises(RuntimeError,match="expected"):
        provisioner.require_frozen_model_2_manifest()
    cfg["training"]["frozen_expected_rows"]=3
    cfg["training"]["frozen_manifest_sha256"]="0"*64
    with pytest.raises(RuntimeError,match="changed"):
        provisioner.require_frozen_model_2_manifest()


def test_calibrated_rf_requires_explicit_group_splits_and_keeps_spectral_input():
    device=DeviceInfo(
        accelerator="cpu",name="test",cpu_count=2,
        torch_available=False,cuda_available=False,mps_available=False,
    )
    alg="random_forest_spatial_calibrated"
    assert get_algorithm(alg).name==alg
    with pytest.raises(ValueError,match="group-isolated"):
        create_model(alg,device,42)
    assert isinstance(create_model(alg,device,42,calibrated=False),
                      RandomForestClassifier)
    groups=np.repeat(np.arange(6),2)
    splits=list(GroupKFold(n_splits=3).split(np.zeros((12,2)),groups=groups))
    assert all(set(groups[tr]).isdisjoint(set(groups[te])) for tr,te in splits)
    calibrated=create_model(
        alg,device,42,
        {"n_estimators":25,"min_samples_leaf":4},
        calibration_cv=splits,
    )
    assert isinstance(calibrated,CalibratedClassifierCV)
    assert calibrated.estimator.min_samples_leaf==4
    assert calibrated.estimator.n_estimators==25
    assert calibrated.cv==splits
