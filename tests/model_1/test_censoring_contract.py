import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_1.steps.step_02_detection_limits import clean_detection_limits
from lithiumscope.model_1.steps.step_06_category_cleaning import clean_categories


def test_censoring_audit_does_not_rewrite_categorical_values():
    raw = pd.DataFrame({"Li_icpms": ["<1e-3", "<0,5", "<=1", "BDL", ">2", "<-2", "<junk"],
                        "Sample_type": ["<2"] * 7, "source_dataset": ["GEOROC"] * 7})
    audit = []
    out = clean_detection_limits(raw, evidence=audit)
    assert out.Li_icpms.iloc[:3].tolist() == pytest.approx([.0005, .25, .5])
    assert np.isnan(out.Li_icpms.iloc[3]) and out.Li_icpms.iloc[4] == 2
    assert out.Li_icpms.iloc[5:].isna().all()
    assert out.Sample_type.tolist() == raw.Sample_type.tolist()
    assert len(audit) == 7 and audit[0]["original"] == "<1e-3"
    assert audit[3]["censor_type"] == "unquantified" and audit[3]["limit"] is None
    assert clean_detection_limits(raw, version=1).Sample_type.iloc[0] == 1


def test_decimal_comma_can_be_disallowed_by_contract():
    out = clean_detection_limits(pd.DataFrame({"Li_icpms": ["<0,5"]}), decimal_comma=False)
    assert out.Li_icpms.isna().all()


def test_semantic_rules_preserve_unknown_and_do_not_match_glacial():
    frame = pd.DataFrame({"Sample_type": ["GL", "WR", "glacial sample", None, "unknown", "volcanic glass"],
                          "Rock_type": ["trachyandesite", "granodiorite", "unknown", None, "unmapped", "basaltic andesite"]})
    out = clean_categories(frame, "common")
    assert out.Sample_type.tolist() == ["volcanic_glass", "whole_rock", "other", "unknown", "unknown", "volcanic_glass"]
    assert out.Rock_type.tolist() == ["volcanic", "plutonic", "unknown", "unknown", "other", "volcanic"]
    assert clean_categories(frame, "common", version=1).Sample_type.iloc[2] == "volcanic_glass"
