"""Directed intake preserves raw originals and does not infer sample counts."""
import pandas as pd
import pytest
from lithiumscope.experiments.data_lab.source_studies.directed_intake import (
    acquire,check_file,inspect,table_inventory)

def test_header_mentions_are_not_sample_counts():
    frame=pd.DataFrame([["Sample","Li (ppm)","Sr"],["A",40,2.3],["B",80,3.5]])
    info,cells=table_inventory(frame,"raw")
    assert info["rows"]==3 and info["li_mentions"]==1
    assert cells[0]["explicit_unit_in_cell"] and cells[0]["column"]==2

def test_invalid_download_content_rejected(tmp_path):
    path=tmp_path/"fake.xls"
    path.write_text("<html>access denied</html>")
    with pytest.raises(ValueError):
        check_file(path)
    with pytest.raises(ValueError):
        acquire(path,"https://example.invalid")
    assert path.read_text()=="<html>access denied</html>"

def test_existing_workbook_preserved_and_inventoried(tmp_path):
    path=tmp_path/"original.xlsx"
    pd.DataFrame([["Li (ppm)",50],["SiO2",70]]).to_excel(path,index=False,header=False)
    before=path.read_bytes()
    assert acquire(path,"https://example.invalid")=="existing"
    sheets,cells,state=inspect(path)
    assert state=="inspected_unmapped" and sheets[0]["li_mentions"]==1
    assert cells[0]["row"]==1 and path.read_bytes()==before

def test_csv_intake_no_assumed_header(tmp_path):
    path=tmp_path/"original.csv"
    path.write_text("sample,Li (ppm)\nA,50\nB,80\n",encoding="utf-8")
    check_file(path)
    sheets,cells,state=inspect(path)
    assert state=="inspected_unmapped" and sheets[0]["rows"]==3
    assert cells[0]["explicit_unit_in_cell"]


def test_selected_source_intake_does_not_reinspect_other_sources(tmp_path,monkeypatch):
    import json
    from lithiumscope.experiments.data_lab.source_studies import directed_intake as intake
    monkeypatch.setattr(intake,"ROOT",tmp_path/"raw")
    monkeypatch.setattr(intake,"RESULTS_DIR",tmp_path/"results")
    out=intake.run(selected=["chen_global_arcs_2020"])
    manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["source_dois"])=={"chen_global_arcs_2020"}
    assert manifest["new_rows_approved"]==0
    assert not (intake.ROOT/"meixner_2020").exists()
    assert pd.read_csv(out/"source_inventory.csv").state.tolist()==["missing"]
    with pytest.raises(ValueError):
        intake.run(selected=["unknown"])


def test_download_alternatives_record_failure_and_success(tmp_path,monkeypatch):
    from lithiumscope.experiments.data_lab.source_studies import directed_intake as intake
    path=tmp_path/"original.xlsx"
    calls=[]
    def fake_acquire(destination,url):
        calls.append(url)
        if url=="first":
            raise ValueError("HTML response")
        pd.DataFrame({"Li":[45.]}).to_excel(destination,index=False)
        return "downloaded"
    monkeypatch.setattr(intake,"acquire",fake_acquire)
    status,url,attempts=intake.download_options(path,("first","second"))
    assert status=="downloaded" and url=="second" and calls==["first","second"]
    assert [a["status"] for a in attempts]==["failed","downloaded"]
    assert attempts[0]["error"]=="HTML response"
    assert intake.download_options(path,("first","second"))==("existing",None,[])
    assert calls==["first","second"]


def test_existing_invalid_original_not_replaced(tmp_path,monkeypatch):
    from lithiumscope.experiments.data_lab.source_studies import directed_intake as intake
    path=tmp_path/"original.xlsx"
    path.write_bytes(b"<html>failed</html>")
    def no_network(*args):
        pytest.fail("Existing file must never be overwritten")
    monkeypatch.setattr(intake,"acquire",no_network)
    status,url,attempts=intake.download_options(path,("first","second"))
    assert status.startswith("existing_invalid:") and url is None and attempts==[]
    assert path.read_bytes()==b"<html>failed</html>"
