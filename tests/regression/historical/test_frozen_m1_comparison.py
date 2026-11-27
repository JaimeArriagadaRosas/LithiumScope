import pandas as pd
import pytest
from lithiumscope.experiments.model_1.evaluation.frozen_comparison import pair_frozen_m1


def fixture():
    return pd.DataFrame({"row_index":[10,11,12],"row_position":[0,1,2],
        "source_dataset":["G"]*3,"Longitude":[-70.,-70.,-71.],
        "Latitude":[-24.,-24.,-25.],"y_true":[20.,20.,30.],
        "spatial_group":["a","a","b"],"outer_fold":[1,1,2],
        "prediction":[19.,21.,29.]})


def test_frozen_identity_keeps_repeated_site_and_lithium():
    old=fixture()
    new=old.iloc[::-1].copy()
    new["prediction"]+=1
    paired,meta=pair_frozen_m1(old,new)
    assert len(paired)==3 and meta["excluded_old"]==0
    assert paired.prediction_new.tolist()==[20.,22.,30.]


@pytest.mark.parametrize("field,value",[("outer_fold",3),("y_true",21.),("Longitude",-69.),
                                          ("row_index",11),("row_position",1)])
def test_identity_mismatch_fails_closed(field,value):
    old=fixture()
    new=old.copy()
    new.loc[0,field]=value
    with pytest.raises(ValueError):
        pair_frozen_m1(old,new)
