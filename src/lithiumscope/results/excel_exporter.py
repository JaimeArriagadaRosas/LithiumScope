from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for raw_name, frame in sheets.items():
            name = raw_name[:31] or "Sheet"
            frame.to_excel(writer, sheet_name=name, index=False)
            worksheet = writer.book[name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                width = min(
                    max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells) + 2,
                    45,
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = max(width, 10)
    return path
