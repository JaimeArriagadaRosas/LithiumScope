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
