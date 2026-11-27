"""Check weighted transfer for other families without changing archived treatment semantics."""
import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.model_1.treatments.transfer_weights import fit_weighted
from lithiumscope.experiments.model_1.studies.expansion.sensitivity import run
from lithiumscope.resources.detection.device import DeviceInfo


@pytest.mark.parametrize("algorithm,params",[
    ("hist_gradient_boosting",{"max_iter":8,"min_samples_leaf":3,"treatment_policy":"rare_group_bounded_log"}),
    ("catboost",{"iterations":8,"depth":3,"treatment_policy":"rare_group_no_coordinates"}),
])
def test_weighted_family_keeps_archived_policy_and_nonnegative_output(algorithm,params):
    if algorithm=="catboost":
        pytest.importorskip("catboost")
    device=DeviceInfo("cpu","test",1,False,False,False)
    index=pd.Index([f"r{i}" for i in range(40)])
    x=pd.DataFrame({"SiO2":np.linspace(50.,75.,40),"Latitude":np.linspace(-40.,-20.,40),
                    "Longitude":np.linspace(-72.,-68.,40),"Rock_type":["volcanic","plutonic"]*20},index=index)
    y=pd.Series(np.linspace(1.,200.,40),index=index)
    groups=pd.Series([f"g{i//4}" for i in range(40)],index=index)
    high=y.ge(180.)
    predict,info=fit_weighted(device,42,params,x,y,groups,["SiO2","Latitude","Longitude"],
                              ["Rock_type"],high,.25,algorithm=algorithm)
    values=predict(x)
    assert values.shape==(40,) and np.isfinite(values).all() and (values>=0).all()
    assert info["algorithm"]==algorithm and info["policy"]==params["treatment_policy"]
    assert info["effective_transfer_weight"]==pytest.approx(.25*info["transfer_fit_rows"])
    assert info["fit_rows"]>len(x)
    if algorithm=="catboost":
        changed=x.assign(Latitude=80.,Longitude=100.)
        assert not info["coordinates"]
        assert np.allclose(values,predict(changed))
    else:
        assert info["bounded"] and info["target_transform"]=="log1p"
        assert values.max()<=y.max()+1e-9


def test_other_families_cannot_run_rf_refinement_or_unregistered_family():
    with pytest.raises(ValueError,match="incompatible"):
        run("missing",allow_unverified_metadata=True,algorithm="catboost",transfer_refinement=True)
    with pytest.raises(ValueError,match="Unsupported"):
        run("missing",allow_unverified_metadata=True,algorithm="svm")
