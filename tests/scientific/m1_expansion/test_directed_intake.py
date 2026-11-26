"""Directed intake preserves raw originals and does not infer sample counts."""
import pandas as pd
import pytest
from lithiumscope.experiments.model_1.augmentation.source_intake import (
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
