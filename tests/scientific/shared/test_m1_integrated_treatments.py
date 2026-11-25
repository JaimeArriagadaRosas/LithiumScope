import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lithiumscope.model_1.training.preprocessing.treatment_policy import TreatedRegressor, rare_group_positions
from lithiumscope.model_1.training.preprocessing.spatial_tabnet import fit_regression_estimator


def test_train_only_bootstrap_and_serialized_predictions(tmp_path):
    import joblib
    x=pd.DataFrame({"a":np.arange(40,dtype=float)},index=[f"row{i}" for i in range(40)])
    y=pd.Series(np.arange(40,dtype=float)**2,index=x.index)
    g=pd.Series(np.repeat(["a","b","c","d"],10),index=x.index)
    model=TreatedRegressor(Pipeline([("scale",StandardScaler()),("model",LinearRegression())]),"rare_group_sqrt",42)
    fit_regression_estimator(model,x,y,g)
    ev=model.treatment_evidence_
    assert ev["real_train_rows"]==40 and ev["fit_rows"]==50
    assert set(ev["copied_train_row_ids"])<=set(x.index)
    assert ev["max_copies_per_parent"]<=4
    test=pd.DataFrame({"a":[-10.,100.]})
    before=model.predict(test)
    assert (before>=0).all()
    path=tmp_path/"model.joblib"
    joblib.dump(model,path)
    assert np.allclose(joblib.load(path).predict(test),before)


def test_bootstrap_matches_verified_experiment():
    from lithiumscope.experiments.generalization.resampling import regression_resample
    y=pd.Series(np.arange(100,dtype=float),index=[str(i) for i in range(100)])
    x=pd.DataFrame({"a":y})
    g=pd.Series(np.repeat(list("abcdefghij"),10),index=y.index)
    _,_,_,trace=regression_resample(x,y,g,g,"rare_group_bootstrap",42)
    assert [str(y.index[i]) for i in rare_group_positions(y,g,42)]==[t["row"] for t in trace]


def test_canonical_factory_produces_nonnegative_raw_schema_predictor():
    from types import SimpleNamespace
    from lithiumscope.model_1.training.factory import build_pipeline
    from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
    x=pd.DataFrame({"a":np.arange(40,dtype=float),"Latitude":np.linspace(-22,-20,40),
                    "Longitude":np.linspace(-70,-68,40)})
    y=pd.Series(np.arange(40,dtype=float))
    g=pd.Series(np.repeat(np.arange(4),10))
    schema=FeatureSchema(list(x.columns),[],2,"stable_v1")
    estimator=build_pipeline("svm",SimpleNamespace(accelerator="cpu"),42,schema,
                             {"treatment_policy":"bounded_log_no_coordinates","C":10.})
    fit_regression_estimator(estimator,x,y,g)
    baseline=estimator.predict(x)
    changed=x.assign(Latitude=0.,Longitude=0.)
    assert np.allclose(baseline,estimator.predict(changed))
    assert ((baseline>=0)&(baseline<=y.max()+1e-10)).all()
