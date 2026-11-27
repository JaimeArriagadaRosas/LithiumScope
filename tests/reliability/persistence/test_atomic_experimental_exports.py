import json
import pandas as pd
import pytest
from lithiumscope.results.atomic_exports import write_json,write_table

def test_atomic_exports_preserve_old_file_on_failure(tmp_path):
    path=tmp_path/"summary.json"
    write_json(path,{"state":"complete"})
    with pytest.raises(ValueError):
        write_json(path,{"invalid":float("nan")})
    assert json.loads(path.read_text())=={"state":"complete"}
    assert list(tmp_path.iterdir())==[path]
    assert pd.read_csv(write_table(tmp_path/"rows.csv",pd.DataFrame({"a":[1]}))).a.tolist()==[1]


def test_historical_export_preserves_index_and_encodes_nonfinite_as_null(tmp_path):
    from lithiumscope.results.atomic_exports import write_legacy_evidence_json,write_text
    path=write_legacy_evidence_json(tmp_path/"historical.json",{"metrics":[1,float("nan"),float("inf")]})
    assert json.loads(path.read_text())=={"metrics":[1,None,None]}
    indexed=write_table(tmp_path/"indexed.csv",pd.DataFrame({"value":[3]},index=pd.Index(["site"],name="region")),index=True)
    assert pd.read_csv(indexed).region.tolist()==["site"]
    assert write_text(tmp_path/"report.md","text\n").read_text()=="text\n"
