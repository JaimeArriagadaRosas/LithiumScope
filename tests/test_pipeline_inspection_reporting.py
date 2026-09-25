import pandas as pd

from lithiumscope.tools.pipeline_inspection_reporting import (
    dataset_summary,
    source_counts,
)


def test_source_counts_returns_named_rows_column():
    frame = pd.DataFrame(
        {
            "source_dataset": [
                "GEOROC",
                "GEOROC",
                "Mamani09 bootstrap",
            ]
        }
    )

    counts = source_counts(frame)

    assert list(counts.columns) == [
        "source_dataset",
        "rows",
    ]
    assert counts.loc[
        counts["source_dataset"] == "GEOROC",
        "rows",
    ].iloc[0] == 2


def test_dataset_summary_handles_missingness_series_reset_index(
    capsys,
):
    frame = pd.DataFrame(
        {
            "source_dataset": [
                "GEOROC",
                "GEOROC",
            ],
            "Li_icpms": [
                10.0,
                20.0,
            ],
            "Longitude": [
                -70.0,
                None,
            ],
            "Latitude": [
                -33.0,
                -34.0,
            ],
        }
    )

    dataset_summary(
        frame,
        "Prueba",
    )

    output = capsys.readouterr().out
    assert "Top 20 columnas" in output
    assert "missing_percent" in output
    assert "Longitude" in output
    assert "Distribución Li_icpms" in output
