from pathlib import Path

import pandas as pd

from lithiumscope.prediction.reporting.maps import (
    save_spatial_maps,
)


def test_spatial_maps_are_generated_from_coordinates(
    tmp_path: Path,
):
    paired = pd.DataFrame(
        [
            {
                "case_id": "a",
                "Longitude": -72.5,
                "Latitude": -44.0,
                "Li_icpms": 20.0,
                "Li_icpms_predicted": 18.0,
                "prospectivity_score": 0.3,
            },
            {
                "case_id": "b",
                "Longitude": -71.0,
                "Latitude": -43.5,
                "Li_icpms": 10.0,
                "Li_icpms_predicted": 12.0,
                "prospectivity_score": 0.7,
            },
        ]
    )

    outputs = save_spatial_maps(
        paired,
        tmp_path,
    )

    assert len(outputs) == 2
    assert all(path.is_file() for path in outputs)
    assert all(path.stat().st_size > 0 for path in outputs)
