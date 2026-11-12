import pandas as pd
import pytest

from lithiumscope.experiments.analysis.georoc_qc_protocols import (
    protocol_masks,
    protocol_summary,
)


def record(**kwargs):
    base = dict(
        SiO2=60, TiO2=1, Al2O3=16, Fe2O3=6,
        MnO=.1, MgO=4, CaO=6, Na2O=3, K2O=3, P2O5=.2,
        Li_icpms=25, Longitude=-70, Latitude=-23,
        Sample_type="WHOLE ROCK", source_dataset="GEOROC",
        Rb_icpms=20, Sr_icpms=100,
    )
    base.update(kwargs)
    return base


def test_A_requires_existing_sum_and_B_accepts_only_full_reconstruction():
    f = pd.DataFrame([record(**{"SUM (no water)": 99.3}),
                      record(**{"SUM (no water)": None}),
                      record(**{"SUM (no water)": None}, Fe2O3=None)])
    m = protocol_masks(f)
    assert m.A_original.tolist() == [True, False, False]
    assert m.B_reconstructed.tolist() == [True, True, False]
    assert m.loc[2, "fully_reconstructed_total"] != m.loc[2, "fully_reconstructed_total"]


def test_C_provisional_requires_analytical_coverage_material_and_provenance():
    f = pd.DataFrame([
        record(**{"SUM (no water)": None}, Fe2O3=None),
        record(**{"SUM (no water)": None}, Fe2O3=None, Sample_type="GLASS"),
        record(**{"SUM (no water)": None}, Fe2O3=None, Li_icpms=None),
        record(**{"SUM (no water)": None}, Fe2O3=None, SiO2=None),
        record(**{"SUM (no water)": None}, Fe2O3=None, MgO=None, P2O5=None),
        record(**{"SUM (no water)": None}, Fe2O3=None, Rb_icpms=None, Sr_icpms=None),
    ])
    m = protocol_masks(f)
    assert m.C_unverified_extra.tolist() == [True, False, False, False, False, False]
    assert m.loc[0, "qc_status"] == "C_UNVERIFIED_EXPERIMENT_ONLY"


def test_C_does_not_invent_or_overwrite_quality_total():
    f = pd.DataFrame([record(**{"SUM (no water)": None}, Fe2O3=None)])
    original = f.copy(deep=True)
    mask = protocol_masks(f)
    assert pd.isna(f.loc[0, "SUM (no water)"])
    pd.testing.assert_frame_equal(f, original)
    assert pd.isna(mask.loc[0, "fully_reconstructed_total"])
    assert mask.loc[0, "C_unverified_extra"]


def test_source_summary_and_invalid_thresholds():
    f = pd.DataFrame([record(**{"SUM (no water)": 99.3}),
                      record(**{"SUM (no water)": None}, Fe2O3=None)])
    mask = protocol_masks(f)
    summary = protocol_summary(f, mask)
    assert len(summary) == 3
    assert summary.set_index("protocol").loc["C_experimental", "additional_vs_A"] == 1
    with pytest.raises(ValueError):
        protocol_masks(f, minimum_majors=0)
