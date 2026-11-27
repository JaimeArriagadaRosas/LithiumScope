"""Protect feature ablation identities, chemistry allowlist and inner selection."""
import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.model_1.studies.expansion.academic_features import (
    ARMS, BASE, measurement_blocks, select_columns, choose, run)


def test_measurements_keep_methods_separate_and_exclude_li():
    x=pd.DataFrame({"Rb_icpms":[10.,20.],"Sr_icpms":[5.,10.]},index=[7,9])
    frame=x.assign(La_icpms=[2.,3.],La_xrf=[4.,5.],Li_icpms=[8.,9.],sample_id=["a","b"])
    recovered=pd.DataFrame({"georoc_La_ppm":[6.,8.],"georoc_Yb_ppm":[2.,4.]},index=x.index)
    measured,ratios=measurement_blocks(frame,recovered,x)
    assert set(measured)=={"La_icpms","La_xrf","georoc_La_ppm","georoc_Yb_ppm"}
    assert np.allclose(ratios["log_ratio_georoc_La_ppm_georoc_Yb_ppm"],np.log([3.,2.]))
    assert np.allclose(ratios["log_ratio_Rb_icpms_Sr_icpms"],np.log(2.))
    with pytest.raises(ValueError):
        measurement_blocks(frame,recovered.assign(georoc_Li_ppm=1.),x)
    with pytest.raises(ValueError):
        measurement_blocks(frame,recovered.iloc[::-1],x)


def test_train_coverage_gate_rejects_empty_constant_and_sparse_group_features():
    g=pd.Series([f"g{i}" for i in range(20)])
    block=pd.DataFrame({"good":np.arange(20.),"constant":1.,"missing":np.nan,
                        "rare":[1.,2.]+[np.nan]*18})
    assert select_columns(block,g)==["good"]
    with pytest.raises(ValueError):
        select_columns(block,g.iloc[::-1])


def test_inner_selection_uses_complete_scores_and_r2_guard():
    scores=pd.DataFrame([{"policy":p,"inner_fold":f,"rmse":4. if p=="combined" else 6.,
        "r2":.4 if p=="combined" else .6} for p in ARMS for f in range(1,4)])
    assert choose(scores)!="combined"
    assert choose(scores.assign(r2=.4))==BASE
    assert choose(scores.assign(r2=.6))=="combined"
    with pytest.raises(ValueError):
        choose(scores.iloc[:-1])


def test_requires_explicit_research_opt_in():
    with pytest.raises(ValueError,match="allow-unverified-metadata"):
        run()
