import pandas as pd
from lithiumscope.results.excel_exporter import export_workbook


def test_excel_exporter_creates_workbook(tmp_path):
    path = tmp_path / "results.xlsx"
    export_workbook(path, {"ranking": pd.DataFrame({"model": ["A"], "rmse": [1.2]})})
    assert path.exists()
    assert path.stat().st_size > 0
