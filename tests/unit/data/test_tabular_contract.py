import pandas as pd
import pytest
from lithiumscope.datasets.readers.tabular import load_data
from lithiumscope.core.exceptions import InputValidationError


@pytest.mark.parametrize("text", ["Li, li\n1,2\n", "Li,,Mg\n1,2,3\n", "Li,Mg\n1,2,3\n", "Li;Mg\n1;2\n", "Li,Mg\n", ""])
def test_ambiguous_and_malformed_tables_are_rejected(tmp_path, text):
    path = tmp_path / "samples.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(InputValidationError):
        load_data(path)


def test_declared_delimiter_preserves_decimal_strings(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("Li;Sample\n1,2;GL\n", encoding="utf-8")
    frame = load_data(path, delimiter=";")
    assert frame.Li.iloc[0] == "1,2"
    assert frame.attrs["tabular_source"]["delimiter"] == ";"


def test_excel_header_and_sheet_contract(tmp_path):
    path = tmp_path / "samples.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Li": [10], "Mg": [2]}).to_excel(writer, sheet_name="chemistry", index=False)
        pd.DataFrame([[1,2]], columns=["Li", " li "]).to_excel(writer, sheet_name="bad", index=False)
    assert load_data(path, sheet_name="chemistry").attrs["tabular_source"]["sheet"] == "chemistry"
    with pytest.raises(InputValidationError, match="ambiguous"):
        load_data(path, sheet_name="bad")
