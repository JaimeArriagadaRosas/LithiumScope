"""Excel sheet selection and header validation."""
import pandas as pd
from lithiumscope.datasets.validation.tabular.schema import validate_headers as _validate_headers


def read_excel(path, sheet_name):
    with pd.ExcelFile(path) as workbook:
        sheet = workbook.sheet_names[sheet_name] if isinstance(sheet_name, int) else sheet_name
        header = pd.read_excel(workbook, sheet_name=sheet, header=None, nrows=1)
        _validate_headers(header.iloc[0].tolist() if not header.empty else [])
        frame = pd.read_excel(workbook, sheet_name=sheet)
    return frame, sheet
