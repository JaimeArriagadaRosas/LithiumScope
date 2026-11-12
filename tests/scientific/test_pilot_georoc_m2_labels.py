"""Explicit opt-in GEOROC labels are never input spectral predictors."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.analysis.pilot_georoc_m2_labels import (
    _quality_of_patch, _stable_sample_id, collect_eligible_label_pairs,
    select_small_pilot,
)


def _fake_frame():
    records=[]
    for i in range(8):
        records.append({
            "source_dataset":"GEOROC",
            "Sample_type":"WHOLE ROCK",
            "SUM (no water)":99.0,
            "SiO2":57.0,"TiO2":1.0,"Al2O3":17.0,"Fe2O3":7.0,
            "MnO":.2,"MgO":5.0,"CaO":6.0,"Na2O":3.0,"K2O":2.0,"P2O5":.8,
            "Li_icpms":10.0 if i%2 else 40.0,
            "Longitude":-74 + i*.6,"Latitude":-20.0,
            "source_sample":f"S{i}","CITATIONS":f"[{i+100}]",
            "Fe_total_provenance":"FEOT_CONVERTED_TO_FE2O3T",
        })
    return pd.DataFrame(records)


def test_explicit_labels_produce_group_and_provenance_without_chemistry_as_input(tmp_path):
    source=tmp_path/"b.csv"
    _fake_frame().to_csv(source,index=False)
    existing=tmp_path/"m2.csv"
    pd.DataFrame({"spatial_group":["g_140_212"]}).to_csv(existing,index=False)
    rows=collect_eligible_label_pairs(source,existing)
    assert len(rows)>=6
    assert "Fe2O3" not in rows and "SiO2" not in rows
    assert "Li_icpms" in rows
    assert "source_citation" in rows
    assert rows.sample_id.str.startswith("GEOROC_").all()
    assert rows.training_eligibility.eq("TECHNICAL_PILOT_ONLY").all()
    chosen=select_small_pilot(rows,limit=4)
    assert len(chosen)==4
    assert chosen.spatial_group.nunique()==4
    assert chosen.source_citation.nunique()==4
    assert chosen.high_li_descriptive.nunique()==2


def test_sample_identifiers_distinguish_publications_and_locations():
    a=_stable_sample_id("[100]","S1",-23,-70)
    b=_stable_sample_id("[101]","S1",-23,-70)
    c=_stable_sample_id("[100]","S1",-23,-70.1)
    assert len({a,b,c})==3


def test_pilot_is_strictly_bounded():
    with pytest.raises(ValueError):
        select_small_pilot(pd.DataFrame(),limit=999)


def test_bad_sentinel_band_is_rejected():
    image=np.ones((6,32,32),dtype=np.float32)
    image[0,:,:]=np.nan
    result=_quality_of_patch(image,("red","green","blue","nir","swir16","swir22"))
    assert not result["technically_usable"]
    assert result["finite_per_band"]["red"]==0
    with pytest.raises(ValueError):
        _quality_of_patch(image[:5],("red","green","blue","nir","swir16","swir22"))


def test_pilot_dry_run_never_contacts_imagery_provider(monkeypatch, tmp_path):
    from lithiumscope.experiments.analysis import pilot_georoc_m2_labels as pilot
    source = tmp_path / "results/scientific_simulations/vnext_20261106/stage2_m1/feot_combined_experiment_only.csv"
    source.parent.mkdir(parents=True)
    _fake_frame().to_csv(source, index=False)
    original = source.read_bytes()
    m2 = tmp_path / "data/processed/model_2/training_manifest.csv"
    m2.parent.mkdir(parents=True)
    pd.DataFrame({"spatial_group":["g_140_213"], "Li_icpms":[21.6]}).to_csv(m2,index=False)
    monkeypatch.setattr(pilot, "load_config", lambda name: {
        "training":{"spatial_group_degrees":.5},
        "imagery":{"normalize_per_band":False}
    })
    monkeypatch.setattr(pilot, "Sentinel2Provider",
        lambda *args, **kwargs: pytest.fail("Dry-run must not access network"))
    report=pilot.run_pilot(root=tmp_path,output=tmp_path/"pilot",limit=4,acquire=False)
    assert report["selected"] == 4
    assert report["patch_valid"] == 0
    assert not (tmp_path/"pilot/patches").exists()
    assert source.read_bytes() == original
