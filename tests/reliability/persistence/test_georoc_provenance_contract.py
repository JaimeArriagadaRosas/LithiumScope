import json
import pandas as pd
from lithiumscope.datasets.adapters.georoc.ingestion import ingest_georoc_files
from lithiumscope.datasets.validation.georoc.source_audit import compare_exports


def test_heterogeneous_files_preserve_units_provenance_and_censoring(tmp_path):
    common = {"MATERIAL": "WHOLE ROCK", "LONGITUDE": -70, "LATITUDE": -20}
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame([{**common, "UNIQUE_ID": "001", "SAMPLE NAME": "A", "CITATIONS": "pub A",
                   "LI(PPM)": "<10", "SIO2(WT%)": 60},
                  {**common, "UNIQUE_ID": "002", "LI(PPM)": "-2", "SIO2(WT%)": 60}]).to_csv(a, index=False)
    pd.DataFrame([{**common, "UNIQUE_ID": "003", "CITATIONS": "pub B", "LI(PPM)": "22", "MGO(WT%)": 4},
                  {**common, "UNIQUE_ID": "004", "MATERIAL": "VOLCANIC GLASS", "LI(PPM)": "20", "MGO(WT%)": 4}]).to_csv(b, index=False)
    result = ingest_georoc_files([a,b], tmp_path/"out.csv", chunksize=1, minimum_predictors=1)
    frame = pd.read_csv(result.path, dtype={"source_unique_id": str})
    assert result.rows_read == 4 and result.rows_kept == 2
    assert result.rows_rejected_missing_core == 2
    assert frame.source_unique_id.tolist() == ["001","003"]
    assert frame.source_publication.tolist() == ["pub A","pub B"]
    assert frame.Li_icpms.tolist() == [5,22]
    assert frame.source_li_raw.tolist() == ["<10","22"]
    assert frame.source_li_unit.tolist() == ["ppm","ppm"]
    assert frame.source_li_method.tolist() == ["unknown","unknown"]
    assert pd.isna(frame.MgO.iloc[0]) and frame.MgO.iloc[1] == 4
    assert frame.SiO2.iloc[0] == 60 and pd.isna(frame.SiO2.iloc[1])
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["analytical_censoring"][0]["original"] == "<10"
    assert audit["rows_skipped_material_core_or_density"] == 2
    assert len(audit["source_sha256"]) == 2


def test_export_audit_detects_chemistry_change(tmp_path):
    raw = pd.DataFrame({"UNIQUE_ID": ["A","B"], "MATERIAL": ["WHOLE ROCK","VOLCANIC GLASS"],
                        "LI(PPM)": [1,2], "CITATIONS": ["p","q"]})
    a, b = tmp_path/"a.csv", tmp_path/"b.csv"
    raw.to_csv(a,index=False)
    raw.iloc[:1].assign(**{"LI(PPM)": [3]}).to_csv(b,index=False)
    audit, _, _ = compare_exports(a,b)
    assert audit["numeric_chemistry_cell_differences"] == 1
    assert not audit["missing_whole_rock_ids"]

def test_integral_id_representation_is_not_a_lost_sample(tmp_path):
    raw = pd.DataFrame({"UNIQUE_ID": ["123", "456"], "MATERIAL": ["WHOLE ROCK", "WHOLE ROCK"],
                        "LI(PPM)": [1,2], "CITATIONS": ["p","q"]})
    a, b = tmp_path/"a.csv", tmp_path/"b.csv"
    raw.to_csv(a,index=False)
    raw.assign(UNIQUE_ID=["123.0", "456.0"]).to_csv(b,index=False)
    audit, _, _ = compare_exports(a,b)
    assert audit["matched_ids"] == 2
    assert audit["numeric_chemistry_cell_differences"] == 0
    assert audit["missing_whole_rock_ids"] == audit["unexpected_filtered_ids"] == []
