import pandas as pd
import pytest

from lithiumscope.core.exceptions import InputValidationError
from lithiumscope.datasets.adapters.georoc.schema import infer_georoc_column_map
from lithiumscope.datasets.adapters.georoc.dataverse import whole_rock_candidates
from lithiumscope.datasets.validation.chemistry.analytical import parse_analytical_value
from lithiumscope.datasets.validation.georoc.dataset_health import assess_dataset_health
from lithiumscope.datasets.validation.target_column import resolve_target


def test_health_distinguishes_presence_from_quantification():
    frame = pd.DataFrame({"LI(PPM)": ["12", "<5", "BDL", "-1", "inf", "unknown"],
                          "LONGITUDE": [-70.]*6, "LATITUDE": [-30.]*6})
    health = assess_dataset_health(frame)
    assert health.li_nonempty == 6
    assert health.li_numeric_valid == 1
    assert health.li_quantified_censored == 1
    assert health.li_invalid_or_unquantified == 4
    assert health.complete_li_coordinate_rows == 2
    with pytest.raises(RuntimeError, match="litio utilizables"):
        assess_dataset_health(frame.iloc[2:])


def test_health_does_not_pick_unrelated_column():
    frame = pd.DataFrame({"LITHOLOGY": ["rock"], "LONGITUDE": [-70.], "LATITUDE": [-30.]})
    with pytest.raises(RuntimeError, match="LI"):
        assess_dataset_health(frame)


def test_ambiguous_alias_and_target_are_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        infer_georoc_column_map(["LI(PPM)", "LI (PPM)"])
    with pytest.raises(InputValidationError, match="ambiguous"):
        resolve_target(pd.DataFrame([[1, 2]], columns=["Li", " li "]), ["Li"])
    with pytest.raises(ValueError, match="lower_fraction"):
        parse_analytical_value("<10", lower_fraction=1.5)


def test_dataverse_uses_exact_material_contract():
    frame = pd.DataFrame({"LATITUDE MIN": [-30.]*3, "LONGITUDE MIN": [-70.]*3,
                         "LI(PPM)": [10, 20, 30], "MATERIAL": ["WR", "WR and GL", "/ WR [123]"],
                         "SAMPLE NAME": ["a", "b", "c"], "CITATIONS": ["x"]*3, "LOCATION": ["x"]*3})
    rows, _ = whole_rock_candidates(frame, "source.csv")
    assert rows.sample_name.tolist() == ["a", "c"]
