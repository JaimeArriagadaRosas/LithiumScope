"""Contracts for enriched synthetic TRAIN data and selection."""
from collections import Counter
import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.model_1.treatments.augmentation import ARMS, augment, MAX_PARENT, MAX_PAIR
from lithiumscope.experiments.model_1.studies.augmentation.academic import choose, run
from lithiumscope.model_1.preparation.feature_engineering import add_geochemical_features
from lithiumscope.experiments.model_1.studies.expansion.academic_features import measurement_blocks


def sample_data():
    idx=pd.Index(["a","b","c","d","e","f"])
    x=pd.DataFrame({"SiO2":[40.,45.,50.,51.,60.,70.],"TiO2":[1.,1.5,2.,2.01,3.,4.],
        "Al2O3":[10.,11.,12.,12.1,14.,15.],"Fe2O3":[3.,4.,5.,5.1,6.,7.],
        "MnO":[.1,.2,.3,.31,.4,.5],"MgO":[1.,2.,3.,3.1,4.,5.],
        "CaO":[2.,3.,4.,4.1,5.,6.],"Na2O":[1.,2.,3.,3.1,4.,5.],
        "K2O":[1.,2.,3.,3.1,4.,5.],"P2O5":[.1,.2,.3,.31,.4,.5],
        "Rb_icpms":[10.,20.,30.,32.,40.,50.],"Sr_icpms":[20.,30.,40.,41.,50.,60.],
        "Latitude":-30.,"Longitude":-70.,"Rock_type":"andesite"},index=idx)
    x=add_geochemical_features(x,version=2)
    recovered=pd.DataFrame({"georoc_La_ppm":[1.,2.,3.,3.1,4.,5.],
                            "georoc_Yb_ppm":[1.,2.,3.,3.2,4.,5.],
                            "georoc_Ce_ppm":[1.,2.,3.,np.nan,4.,5.]},index=idx)
    measured,ratios=measurement_blocks(x,recovered,x)
    y=pd.Series([5.,10.,50.,60.,15.,20.],index=idx)
    groups=pd.Series("g",index=idx)
    sources=pd.Series("GEOROC",index=idx)
    return pd.concat([x,measured,ratios],axis=1),y,groups,sources,list(x),list(measured),["Rock_type"]


@pytest.mark.parametrize("policy",["smoter_enriched","gaussian_enriched"])
def test_synthetic_ratios_recomputed_and_no_new_sites(policy):
    args=sample_data()
    x,y,_,_,_,_,_=args
    xx,yy,w,trace,info=augment(*args,policy,42)
    assert info["compatible_pairs"]==1 and info["added_rows"]==MAX_PAIR
    assert len(xx)==len(yy)==len(w)==len(x)+len(trace)
    assert set(yy.iloc[len(x):].between(50,60))=={True}
    generated=xx.iloc[len(x):]
    assert generated.Latitude.eq(-30.).all() and generated.Longitude.eq(-70.).all()
    assert generated.georoc_Ce_ppm.isna().all()
    assert generated.georoc_La_ppm.between(3.,3.1).all()
    assert np.allclose(generated["log_ratio_georoc_La_ppm_georoc_Yb_ppm"],
                       np.log(generated.georoc_La_ppm/generated.georoc_Yb_ppm))
    assert np.allclose(generated.Alkali_Sum,generated.Na2O+generated.K2O)
    assert np.allclose(generated.Mg_Number,generated.MgO/(generated.MgO+generated.Fe2O3))
    assert np.allclose(generated["log_ratio_Rb_icpms_Sr_icpms"],np.log(generated.Rb_icpms/generated.Sr_icpms))
    counts=Counter(row[key] for row in trace for key in ("parent_a","parent_b"))
    assert set(counts)=={"c","d"} and max(counts.values())<=MAX_PARENT
    second=augment(*args,policy,42)
    pd.testing.assert_frame_equal(xx,second[0])
    assert trace==second[3]


@pytest.mark.parametrize("field",["source","group","site","category"])
def test_incompatible_rare_parents_do_not_force_generation(field):
    x,y,g,s,canonical,measured,categorical=sample_data()
    if field=="source":
        s.loc["d"]="Mamani09"
    elif field=="group":
        g.loc["d"]="different"
    elif field=="site":
        x.loc["d","Latitude"]=-30.1
    else:
        x.loc["d","Rock_type"]="rhyolite"
    xx,yy,w,trace,info=augment(x,y,g,s,canonical,measured,categorical,"smoter_enriched",42)
    assert not trace and info["compatible_pairs"]==0 and info["added_rows"]==0
    pd.testing.assert_frame_equal(xx,x)
    pd.testing.assert_series_equal(yy,y)


def test_copies_and_density_controlled_budget():
    args=sample_data()
    for policy in ("rare_copies","relevance_copies"):
        xx,yy,w,trace,info=augment(*args,policy,42)
        assert info["added_rows"]<=4 and len(trace)==info["added_rows"]
        assert max(Counter(row["parent_a"] for row in trace).values())<=MAX_PARENT
        assert all(row["action"]=="copy" for row in trace)
        assert np.allclose(w,1.)
    xx,yy,w,trace,info=augment(*args,"density_kde",42)
    assert not trace and len(xx)==6 and np.isclose(w.mean(),1.)
    assert (w>0).all()
    with pytest.raises(ValueError):
        augment(args[0],args[1].iloc[::-1],*args[2:],"control",42)


def test_inner_selection_requires_improvement_and_r2_guard():
    rows=pd.DataFrame([{"policy":p,"inner_fold":i,"rmse":6. if p=="control" else 5.,
                       "r2":.4 if p=="gaussian_enriched" else .6}
                       for p in ARMS for i in range(1,4)])
    assert choose(rows) not in {"control","gaussian_enriched"}
    assert choose(rows.assign(rmse=6.))=="control"
    with pytest.raises(ValueError):
        choose(rows.iloc[:-1])
    with pytest.raises(ValueError,match="allow-unverified-metadata"):
        run()
