"""Coordinate coalescence and source bounding-box diagnostics for M2 eligibility."""
import numpy as np
import pandas as pd

from lithiumscope.experiments.analysis.audit_georoc_m2_candidate_coverage import (
    coalesced_coordinates,coordinate_half_extent_m,
)


def test_per_row_coordinates_recover_georoc_where_old_merged_columns_are_empty():
    d=pd.DataFrame({
        "Longitude (X)":[-73.0,np.nan],
        "Latitude (Y)":[-20.0,np.nan],
        "Longitude":[np.nan,-68.0],
        "Latitude":[np.nan,-25.0],
    })
    out=coalesced_coordinates(d)
    assert out.longitude.tolist()==[-73.0,-68.0]
    assert out.latitude.tolist()==[-20.0,-25.0]


def test_coordinate_radius_matches_bounds_but_does_not_infer_accuracy():
    d=pd.DataFrame({
        "Latitude_min":[-23.,-23.,np.nan],
        "Latitude_max":[-23.,-22.98,np.nan],
        "Longitude_min":[-70.,-70.,-70.],
        "Longitude_max":[-70.,-69.98,-70.],
    })
    radius=coordinate_half_extent_m(d)
    assert radius.iloc[0]==0
    assert radius.iloc[1]>320
    assert pd.isna(radius.iloc[2])
