"""Physical-label provenance checks; synthetic fixtures, no web or real fitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from lithiumscope.model_2.methodology.contracts import sample_contract
from lithiumscope.model_2.methodology.physical_label_audit import (
    conflicting_site_dossier, georoc_links, historical_links, patch_quality,
    safe_sample_id,
)


@pytest.fixture
def raw_mamani():
    return pd.DataFrame({
        "Sample":[" A 1","B","C"],
        "Li_icpms":[12.,33.,18.],
        "Latitude (Y)":[-23.,-23.,-24.],
        "Longitude (X)":[-70.,-70.,-71.],
        "Geochemistry Reference":["Mamani et al., 2008"]*3,
        "Sample_type":["lava","intrusion","lava"],
        "Rock_type":["andesite","granite","basalt"],
        "Country":["Chile"]*3,
        "Location":["Site A","Site A","Site B"],
        "Geologycal_age":["Quaternary"]*3,
    })


@pytest.fixture
def sample_contract_fixture():
    return sample_contract(pd.DataFrame({
        "sample_id":["A_1","B","C"],
        "image_path":["a.npy","b.npy","c.npy"],
        "Li_icpms":[12.,33.,18.],
        "latitude":[-23.,-23.,-24.],
        "longitude":[-70.,-70.,-71.],
        "spatial_group":["x","x","z"],
        "sentinel_scene_id":["S2_A","S2_A","S2_C"],
        "sentinel_cloud_cover":[0.,0.,0.],
        "scl_cloud_fraction":[0.,0.,0.],
    }))


def test_source_sample_normalization_preserves_identity():
    assert safe_sample_id(" A 1")=="A_1"
    assert safe_sample_id("A-12")=="A-12"
    assert safe_sample_id(None)==""


def test_historical_join_checks_li_and_coordinates(sample_contract_fixture,raw_mamani):
    source=raw_mamani.copy()
    contract=sample_contract_fixture.copy(deep=True)
    before=contract.copy(deep=True)
    result=historical_links(contract,source)
    assert result.source_link_status.eq("SOURCE_ROW_CROSSCHECKED").all()
    assert result.geochemistry_reference.eq("Mamani et al., 2008").all()
    assert result.raw_rock_type.tolist()==["andesite","granite","basalt"]
    assert_frame_equal(contract,before)
    mismatch=source.copy()
    mismatch.loc[0,"Li_icpms"]=100.
    bad=historical_links(contract,mismatch)
    assert bad.loc[0,"source_link_status"]=="SOURCE_ROW_NOT_FOUND"
    assert bad.loc[1,"source_link_status"]=="SOURCE_ROW_CROSSCHECKED"
    mismatch=source.copy()
    mismatch.loc[0,"Latitude (Y)"]=-22.
    bad=historical_links(contract,mismatch)
    assert bad.loc[0,"source_link_status"]=="SOURCE_ROW_NOT_FOUND"


def test_ambiguous_raw_rows_are_not_certified(sample_contract_fixture,raw_mamani):
    duplicate=pd.concat([raw_mamani,raw_mamani.iloc[[0]]],ignore_index=True)
    matched=historical_links(sample_contract_fixture,duplicate)
    assert matched.loc[0,"source_link_status"]=="MULTIPLE_RAW_ROWS"
    assert matched.loc[0,"raw_row_matches"]==2


def test_georoc_provisional_and_raw_reference_does_not_certify_field():
    item=sample_contract(pd.DataFrame({
        "sample_id":["GEOROC_1"],
        "image_path":["patch.npy"],
        "Li_icpms":[32.],
        "latitude":[-26.5],
        "longitude":[-70.5],
        "spatial_group":["g1"],
        "source_citation":["[123]"],
        "source_dataset":["GEOROC_B_SURFACE_PROVISIONAL"],
        "coordinate_verification":["PRE_SCREENED_NOT_FIELD_VERIFIED"],
    }))
    source=pd.DataFrame([{
        "sample_id":"GEOROC_1","CITATIONS":"[123]",
        "source_sample":"original_1","Latitude":-26.5,"Longitude":-70.5,
        "Li_icpms":32.,"raw_rock_class":"/ NOT GIVEN [123]",
        "raw_comment":"","decimal_lat":3,"decimal_lon":5,
        "references_complete":True,"screening_tier":"PROVISIONAL_SITE_REVIEW",
        "geologic_label_certified":False,"source_dois":"10.1000/test",
        "raw_matches":1,"LOCATION":"CHILE",
    }])
    bib=pd.DataFrame([{"ref_id":"123","matched":True,
                       "citation_text":"Original research","doi":"10.1000/test"}])
    queue=pd.DataFrame({"sample_id":["GEOROC_1"]})
    result=georoc_links(item,source,bib,queue)
    assert result.reference_present_in_official_snapshot.all()
    assert result.rock_description_missing.all()
    assert result.nominal_coordinate_precision_low.all()
    assert not result.sample_to_patch_certified.any()
    assert not result.field_gps_verified.any()
    bad=source.copy()
    bad.loc[0,"Longitude"]=-70.6
    with pytest.raises(ValueError,match="disagree"):
        georoc_links(item,bad,bib,queue)
    bad=source.copy()
    bad.loc[0,"screening_tier"]="MANUAL_SOURCE_REVIEW"
    with pytest.raises(ValueError,match="not provisional"):
        georoc_links(item,bad,bib,queue)
    bad=bib.copy()
    bad.loc[0,"matched"]=False
    with pytest.raises(ValueError,match="bibliography"):
        georoc_links(item,source,bad,queue)


def test_patch_quality_reports_finite_pixels_and_no_automatic_rock_certification(
    tmp_path, sample_contract_fixture
):
    img=tmp_path/"A_1.npy"
    x=np.ones((6,32,32),dtype=np.float32)
    x[0]=.2
    x[3]=.5
    np.save(img,x,allow_pickle=False)
    meta={"sample_id":"A_1","latitude":-23.,"longitude":-70.,
          "scene_id":"S2_A","bands":["red","green","blue","nir","swir16","swir22"]}
    img.with_suffix(".json").write_text(json.dumps(meta),encoding="utf-8")
    row=sample_contract_fixture.iloc[0].copy()
    row["image_path"]=str(img)
    good=patch_quality(row)
    assert good["patch_integrity"]=="OK"
    assert good["finite_pixel_fraction"]==1
    assert good["ndvi_median_proxy"]>0.4
    assert good["scl_rock_exposure_certified"] is False
    x[:,:10,:10]=np.nan
    np.save(img,x,allow_pickle=False)
    bad=patch_quality(row)
    assert bad["patch_integrity"]=="INSUFFICIENT_FINITE_PIXELS"
    assert bad["finite_pixel_fraction"]<.95
    meta["scene_id"]="S2_BAD"
    img.with_suffix(".json").write_text(json.dumps(meta),encoding="utf-8")
    bad=patch_quality(row)
    assert bad["patch_integrity"]=="BAD_SIDECAR_METADATA"
    assert not bad["meta_scene_matches"]


def test_identical_spectral_patch_with_contradictory_li_is_review_not_auto_relabel(
    sample_contract_fixture,raw_mamani
):
    sub=sample_contract_fixture.iloc[:2].copy()
    sub["site_has_conflicting_labels"]=True
    old=historical_links(sub,raw_mamani)
    old["image_sha256"]=["same_image_hash","same_image_hash"]
    report=conflicting_site_dossier(sub,old)
    assert len(report)==1
    assert report.iloc[0].n_samples==2
    assert report.iloc[0].high_count_at_21p6==1
    assert report.iloc[0].low_count_at_21p6==1
    assert report.iloc[0].exactly_same_image_bytes
    assert report.iloc[0].raw_rock_type_count==2
    assert not report.iloc[0].field_verified
    assert report.iloc[0].review_priority.startswith("P0")


def test_missing_patch_is_reported_without_writing_original(tmp_path,sample_contract_fixture):
    row=sample_contract_fixture.iloc[0].copy()
    row["image_path"]=str(tmp_path/"missing.npy")
    output=patch_quality(row)
    assert output["patch_integrity"]=="MISSING_IMAGE_OR_METADATA"
    assert not output["image_available"]
    assert not (tmp_path/"missing.npy").exists()


from lithiumscope.model_2.methodology.physical_label_audit import scientific_readiness_gate


def test_scientific_gate_blocks_uncertified_physical_label_without_deleting_rows():
    summary={
        "historical_conflicting_sites":25,
        "patch_integrity_failures":16,
        "historical_source_status":{"MULTIPLE_RAW_ROWS":1},
        "georoc_77_coordinate_precision_3_or_fewer_decimals":25,
        "samples_with_auto_field_certification":0,
    }
    result=scientific_readiness_gate(summary)
    assert result["scientific_certification_pass"] is False
    assert len(result["rejection_reasons"])==5
    assert result["automatically_excluded"]==0
    assert result["training_executed"] is False
    assert result["verified_surface_labels"]==0
