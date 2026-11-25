"""Regression probes for the new isolated M1 experimental treatments."""
import numpy as np
import pandas as pd

from lithiumscope.experiments.model_1.targeted.features import FoldFeatures, add_ratios
from lithiumscope.experiments.model_1.targeted.policies import ALGORITHMS, ARMS, Arm, Target, adaptive_positions, weights
from lithiumscope.experiments.model_1.targeted.runner import select_arm


def test_predeclared_configurations():
    assert len(ALGORITHMS)*len(ARMS) == 45
    assert len({a.name for a in ARMS}) == len(ARMS)


def test_weights_and_quantile_copies_use_only_train():
    y = np.r_[np.arange(100), 696.]
    groups = np.repeat(np.arange(101)//5, 1)
    sources = np.where(groups < 10, "a", "b")
    for mode in ("none","density_05","density_10","source_group"):
        result = weights(y,groups,sources,mode)
        assert np.isfinite(result).all() and (result > 0).all()
        np.testing.assert_allclose(result.mean(),1.)
    extra = adaptive_positions(y,groups,42)
    assert (extra < len(y)).all()
    assert pd.Series(extra).value_counts().max() <= 4
    assert (y[extra] >= np.quantile(y,.75)).all()


def test_target_roundtrip():
    y = np.array([0.,1.,10.,696.])
    for name in ("identity","sqrt","power075","yeo_johnson"):
        target = Target(name).fit(y)
        np.testing.assert_allclose(target.decode(target.encode(y)),y,rtol=1e-8,atol=1e-8)


def test_ratios_missing_and_conditioned_imputation():
    x = pd.DataFrame({"Rb_icpms":[1.,2.,3.,4.,5.,np.nan],"Sr_icpms":[2.,2.,2.,2.,2.,0.],
                      "Rock_type":["a"]*6})
    assert np.isnan(add_ratios(x).iloc[-1]["log_ratio_Rb_icpms_Sr_icpms"])
    ff = FoldFeatures(Arm("impute",impute=True),["Rb_icpms","Sr_icpms"],["Rock_type"])
    ff.fit(x,pd.Series(np.arange(6)),pd.Series(np.arange(6)))
    other = pd.DataFrame({"Rb_icpms":[np.nan,np.nan],"Sr_icpms":[np.nan,np.nan],"Rock_type":["a","b"]})
    assert ff.transform(other).Rb_icpms.iloc[0] == 3.
    assert np.isnan(ff.transform(other).Rb_icpms.iloc[1])


def test_selection_preserves_mae_and_uses_inner_scores():
    rows = [{"policy":"control","rmse":10.,"mae":5.,"high_bias":-2.},
            {"policy":"bad_mae","rmse":5.,"mae":6.,"high_bias":0.},
            {"policy":"good","rmse":9.,"mae":4.,"high_bias":-1.}]
    assert select_arm(rows) == "good"


def test_fit_weighted_rf_smoke():
    from lithiumscope.experiments.model_1.targeted.fit import fit_arm
    from lithiumscope.resources.detection.device import detect_device
    rng = np.random.default_rng(42)
    x = pd.DataFrame({"Rb_icpms":rng.uniform(1.,100.,60),"Sr_icpms":rng.uniform(1.,50.,60)})
    y = pd.Series(np.maximum(0.,x.Rb_icpms*.5+rng.normal(0.,2.,60)),index=x.index)
    groups = pd.Series(np.arange(60)//5,index=x.index)
    sources = pd.Series("synthetic-test",index=x.index)
    test = pd.DataFrame({"Rb_icpms":[10.,50.],"Sr_icpms":[2.,4.]},index=[60,61])
    scores,train,evidence = fit_arm("random_forest",Arm("weighted",weight="density_05",transform="power075"),
        detect_device(prefer_gpu=False),42,{"n_estimators":5,"max_depth":3},
        x,y,groups,sources,list(x.columns),[],test)
    assert scores.shape == (2,) and train.shape == (60,)
    assert np.isfinite(scores).all() and (scores>=0).all()
    assert evidence["fit_rows"] == len(x)
