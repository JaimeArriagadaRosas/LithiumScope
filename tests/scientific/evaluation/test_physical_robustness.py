import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_1.diagnostics.robustness import apply_stress,evaluate_stress,SCENARIOS,SEED_OFFSETS
from lithiumscope.model_1.preparation.feature_engineering import add_geochemical_features
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema,OXIDES,DERIVED


def sample():
    frame=pd.DataFrame({
        "SiO2":[60.,50.,60.,101.],"TiO2":[.8,.8,.9,.5],"Al2O3":[15.,20.,15.,14.],
        "Fe2O3":[8.,7.,5.,6.],"MnO":[.2,.3,.1,.1],"MgO":[3.,5.,0.,3.],
        "CaO":[5.,6.,4.,5.],"Na2O":[4.,3.,5.,4.],"K2O":[2.,3.,2.,2.],
        "P2O5":[.2,np.nan,.3,.2],"Th_icpms":[1.,2.,np.nan,-1.],"Rb_icpms":[10.,20.,30.,40.],
        "Latitude":[-30.,90.,np.nan,-25.],"Longitude":[-70.,179.999,50.,-69.],"Rock_type":["a","b","c","a"]},
        index=[12,40,45,88])
    frame=add_geochemical_features(frame,version=2)
    schema=FeatureSchema([c for c in frame if c!="Rock_type"],["Rock_type"],2)
    return frame,schema


def test_oxide_perturbation_preserves_observed_valid_oxide_sum_and_derived():
    x,schema=sample()
    original=x.copy(deep=True)
    stress=apply_stress(x,schema,"oxide_relative",.05,42)
    out=stress.frame
    pd.testing.assert_frame_equal(out,apply_stress(x,schema,"oxide_relative",.05,42).frame)
    pd.testing.assert_frame_equal(x,original)
    assert out.index.equals(x.index) and out.columns.equals(x.columns)
    pd.testing.assert_frame_equal(out[["Latitude","Longitude","Rock_type","Th_icpms","Rb_icpms"]],
                                  x[["Latitude","Longitude","Rock_type","Th_icpms","Rb_icpms"]])
    assert out.loc[88,"SiO2"]==101. and stress.preexisting_invalid_cells==1
    np.testing.assert_allclose(out[OXIDES].sum(axis=1),x[OXIDES].sum(axis=1),atol=1e-10)
    assert out[OXIDES].isna().equals(x[OXIDES].isna())
    pd.testing.assert_frame_equal(out[DERIVED],add_geochemical_features(out,version=2)[DERIVED])
    assert stress.changed_cells>0
    assert out.loc[45,"MgO"]==0. and np.isnan(out.loc[45,"K_Mg_ratio"])


def test_trace_noise_only_raw_trace_and_missingness_recomputes_derived():
    x,schema=sample()
    trace=apply_stress(x,schema,"trace_relative",.05,55)
    untouched=[c for c in x if c not in {"Th_icpms","Rb_icpms"}]
    pd.testing.assert_frame_equal(trace.frame[untouched],x[untouched])
    assert trace.frame.loc[88,"Th_icpms"]==-1.
    assert trace.preexisting_invalid_cells==1
    for col in ("Th_icpms","Rb_icpms"):
        mask=x[col].gt(0)
        assert np.all(np.abs(trace.frame.loc[mask,col]/x.loc[mask,col]-1)<=.05+1e-12)
    missing=apply_stress(x,schema,"missingness",.15,42)
    assert missing.changed_cells>0
    assert missing.frame[OXIDES].isna().sum().sum()>=x[OXIDES].isna().sum().sum()
    pd.testing.assert_frame_equal(missing.frame[DERIVED],add_geochemical_features(missing.frame,version=2)[DERIVED])


def test_spatial_shift_geodesic_and_chemistry_untouched():
    from sklearn.metrics.pairwise import haversine_distances
    x,schema=sample()
    out=apply_stress(x,schema,"spatial_m",500.,42).frame
    keep=x[["Latitude","Longitude"]].notna().all(axis=1)
    distances=haversine_distances(np.deg2rad(x.loc[keep,["Latitude","Longitude"]]),
                                  np.deg2rad(out.loc[keep,["Latitude","Longitude"]])).diagonal()*6371008.8
    np.testing.assert_allclose(distances,500,atol=.001)
    pd.testing.assert_frame_equal(x.drop(columns=["Latitude","Longitude"]),out.drop(columns=["Latitude","Longitude"]))
    assert out.Longitude.between(-180,180).all() and out.Latitude.dropna().between(-90,90).all()


def test_paired_stress_records_failure_and_never_fits():
    x,schema=sample()
    class Predictor:
        def fit(self,*args):
            raise AssertionError("Stress must not fit")
        def predict(self,frame):
            return frame.SiO2.to_numpy(float) if frame.SiO2.notna().all() else np.full(len(frame),np.nan)
    base=x.SiO2.to_numpy(float)
    y=base-1
    rows=evaluate_stress(Predictor(),x,y,base,schema,fold=2,seed=42)
    assert len(rows)==sum(map(len,SCENARIOS.values()))*len(SEED_OFFSETS)==16
    assert set(row["status"] for row in rows)<={"ok","failed"}
    assert all(row["n_test"]==len(x) and row["fold"]==2 for row in rows)
    assert any(r["status"]=="ok" and r["scenario"]=="oxide_relative" for r in rows)
    with pytest.raises(ValueError,match="predeclared"):
        apply_stress(x,schema,"spatial_m",123.,42)


def test_no_features_explicitly_unavailable():
    x=pd.DataFrame({"x":[1.,2.]})
    schema=FeatureSchema(["x"],[],2)
    rows=evaluate_stress(None,x,[1.,2.],np.array([1.,2.]),schema,fold=1,seed=42)
    assert len(rows)==16 and all(row["status"]=="unavailable" for row in rows)
