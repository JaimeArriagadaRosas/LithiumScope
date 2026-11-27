"""GEOROC research orchestration tests; no real network, model training or archives."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lithiumscope.datasets.management.snapshots import verify_snapshot
from lithiumscope.experiments.georoc.imagery.coordinate_plan import coordinate_plan
from lithiumscope.experiments.georoc.imagery import batch
from lithiumscope.experiments.georoc.provenance.bibliography import unambiguous_references
from lithiumscope.experiments.georoc.workflow import run
from lithiumscope.experiments.scenarios.imagery import image_catalog
from lithiumscope.tools.acquisition.publications import lookup_doi, fetch_public_pdf


def _candidates():
    return pd.DataFrame({"candidate_gate":["REVIEW_METHOD_AND_SURFACE"]*4,
        "nearest_full_mamani_km":[200,300,20,300],
        "latitude":[-30.,-30.00001,-25.,-100.],"longitude":[-70.,-70.,-72.,-70.]})


def test_coordinate_plan_is_stable_and_retains_excluded_rows():
    source=_candidates()
    sites,ledger=coordinate_plan(source)
    assert len(sites)==1 and len(ledger)==4
    assert ledger.plan_state.tolist()==["selected","selected","not_selected","not_selected"]
    changed=pd.concat([source.iloc[::-1],pd.DataFrame({"candidate_gate":["REVIEW_METHOD_AND_SURFACE"],
        "nearest_full_mamani_km":[400],"latitude":[-40.],"longitude":[-75.]})],ignore_index=True)
    newer,_=coordinate_plan(changed)
    assert sites.iloc[0].image_id in newer.image_id.tolist()
    assert coordinate_plan(source.iloc[0:0])[0].empty


def test_citations_with_conflicting_metadata_are_not_resolved():
    refs=pd.DataFrame({"CITATIONS":[1,1,2,2],"DOI":["x","y","z","z"]})
    resolved,ambiguous=unambiguous_references(refs)
    assert ambiguous=={1} and resolved.citation_number.tolist()==[2]


def test_local_pipeline_reuses_verified_stages(tmp_path):
    source=tmp_path/"part.csv"
    raw=pd.DataFrame({"LATITUDE MIN":[-30,-31,-31],"LONGITUDE MIN":[-70,-71,-71],
      "LI(PPM)":[10,20,20],"MATERIAL":["WR"]*3,"SAMPLE NAME":["A","B","B"],
      "CITATIONS":["[1]","[2]","[2]"],"LOCATION":["ARC / CHILE"]*3,
      "UNIQUE_ID":["a","b","c"],"DRILL DEPTH MIN":[0]*3,"ROCK TYPE":["VOL"]*3,
      "LOCATION COMMENT":["surface"]*3})
    raw.to_csv(source,index=False)
    training=tmp_path/"training.csv"
    pd.DataFrame({"latitude":[-10.],"longitude":[-60.]}).to_csv(training,index=False)
    mamani=tmp_path/"mamani.csv"
    pd.DataFrame({"Latitude (Y)":[-10.]*701,"Longitude (X)":[-60.]*701,"Li_icpms":[5.]*701,
      "Sample":[str(i) for i in range(701)],"Geochemistry Reference":["ref"]*701}).to_csv(mamani,index=False)
    citations=tmp_path/"citations.tab"
    pd.DataFrame({"CITATIONS":[1,2],"DOI":["x","y"],"TITLE":["A","B"],"AUTHORS":["A","B"],
      "YEAR":[2020,2021],"FORMATTED CITATION":["a","b"]}).to_csv(citations,sep="\t",index=False)
    root=tmp_path/"results"
    inputs=[source,training,mamani,citations]
    before=[p.read_bytes() for p in inputs]
    assert run([source],training,mamani,citations,root=root)["state"]=="preflight"
    assert not root.exists()
    first=run([source],training,mamani,citations,root=root,execute=True)
    assert first["state"]=="complete" and len(first["stages"])==5
    frame=pd.read_csv(first["report"])
    assert frame.scientific_validation_approved.eq(False).all()
    assert set(frame.candidate_gate)=={"REVIEW_METHOD_AND_SURFACE","PRIMARY_ROW_UNRESOLVED"}
    assert [p.read_bytes() for p in inputs]==before
    again=run([source],training,mamani,citations,root=root,execute=True)
    assert all(v["reused"] for v in again["stages"].values())
    for v in again["stages"].values():
        verify_snapshot(Path(v["directory"]))


def test_imagery_preflight_does_not_load_model_or_provider(tmp_path,monkeypatch):
    source=tmp_path/"candidates.csv"
    _candidates().to_csv(source,index=False)
    monkeypatch.setattr(batch,"latest_model_path",lambda *a:pytest.fail("preflight loaded model"))
    out=tmp_path/"results"
    result=batch.run(source,root=out,provider_factory=lambda *a:pytest.fail("preflight contacted provider"))
    assert result["sites"]==1 and not out.exists()


def test_site_cache_checks_all_artifacts_and_recovers(tmp_path,monkeypatch):
    calls=[]
    def fake_patch(identity,lat,lon,path,provider,signature):
        calls.append(identity)
        for p in (path,path.with_suffix(".json"),path.with_suffix(".scl.npy"),path.with_suffix(".scene-audit.json")):
            p.write_bytes(b"fixture")
        return path,{"scene_id":"scene"}
    def fake_score(model,path,site,seed):
        out=site/"image_scenario_metrics.csv"
        pd.DataFrame({"scenario":[s.name for s in image_catalog(seed)],"status":"SCORED","score":.5}).to_csv(out,index=False)
        return out
    monkeypatch.setattr(batch,"download_registered_patch",fake_patch)
    monkeypatch.setattr(batch,"evaluate_verified_image",fake_score)
    row=SimpleNamespace(image_id="coord_example",latitude=-30.,longitude=-70.)
    site=tmp_path/"sites"/row.image_id
    first=batch._site(site,row,{},None,"plan",42,recover=False)
    assert batch._site(site,row,{},None,"plan",42,recover=False)==first and len(calls)==1
    (site/"patch.scl.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="integrity"):
        batch._site(site,row,{},None,"plan",42,recover=False)
    batch._site(site,row,{},None,"plan",42,recover=True)
    assert len(calls)==2 and len(list((tmp_path/".quarantine").iterdir()))==1


def test_publication_preflight_and_doi_identity(tmp_path):
    def forbidden(*a,**k):
        pytest.fail("Unexpected network")
    assert fetch_public_pdf("https://example.invalid",tmp_path/"a.pdf",request=forbidden)["state"]=="preflight"
    assert lookup_doi("10.1234/example",request=forbidden)["state"]=="preflight"
    response=SimpleNamespace(raise_for_status=lambda:None,json=lambda:{"doi":"https://doi.org/10.1234/other"})
    with pytest.raises(ValueError,match="identity"):
        lookup_doi("10.1234/example",acquire=True,request=lambda *a,**k:response)
