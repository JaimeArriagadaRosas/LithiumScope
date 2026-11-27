import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema, build_preprocessor
from lithiumscope.model_1.training.preprocessing.evidence import preprocessing_evidence
from lithiumscope.model_1.training.preprocessing.variants import feature_variant, ABLATIONS
from lithiumscope.model_1.preparation.training_data import PreparedModel1Data


def test_fold_empty_feature_and_unseen_categories_are_explicit():
    train=pd.DataFrame({"SiO2":[50.,60.,70.],"MgO":[np.nan]*3,"Rock_type":["volcanic",None,"plutonic"]})
    test=pd.DataFrame({"SiO2":[999.,-999.],"MgO":[2.,np.nan],"Rock_type":["new",None]})
    schema=FeatureSchema(["SiO2","MgO"],["Rock_type"],2,"stable_v1")
    estimator=Pipeline([("preprocess",build_preprocessor(schema,"svm")),("model",DummyRegressor())])
    estimator.fit(train,[1.,2.,3.])
    prep=estimator.named_steps["preprocess"]
    names=list(prep.get_feature_names_out())
    before=pickle.dumps(prep)
    t=prep.transform(test)
    assert pickle.dumps(prep)==before
    assert t.shape[1]==len(names)==6
    assert "numeric__MgO" in names
    assert t[0,names.index("categorical__Rock_type___lithiumscope_unseen__")]==1
    assert t[1,names.index("categorical__Rock_type___lithiumscope_missing__")]==1
    evidence=preprocessing_evidence(estimator,train,test)
    assert evidence["train_all_missing_columns"]==["MgO"]
    assert evidence["transformers"]["numeric"]["scaler_mean_"]==[60.,0.]
    assert evidence["transformers"]["categorical"]["unseen_test_counts"]=={"Rock_type":1}
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(estimator)).predict(test),estimator.predict(test))


def test_tabnet_mask_distinguishes_zero_from_missing():
    x=pd.DataFrame({"SiO2":[0.,np.nan,1.],"MgO":[np.nan]*3,"Rock_type":[None]*3})
    schema=FeatureSchema(["SiO2","MgO"],["Rock_type"],2,"tabnet_mask_v1")
    prep=build_preprocessor(schema,"tabnet").fit(x)
    t=prep.transform(x)
    names=list(prep.get_feature_names_out())
    assert t[0,names.index("numeric__SiO2")]==t[1,names.index("numeric__SiO2")]==0
    assert t[0,names.index("missing_mask__missingindicator_SiO2")]==0
    assert t[1,names.index("missing_mask__missingindicator_SiO2")]==1
    assert t[:,names.index("missing_mask__missingindicator_MgO")].tolist()==[1.,1.,1.]
    with pytest.raises(ValueError,match="only valid"):
        build_preprocessor(schema,"svm")


def test_ablations_preserve_rows_targets_and_groups():
    from lithiumscope.core.pipeline_audit import PipelineAudit
    x=pd.DataFrame({"SiO2":[1.,2.],"Longitude":[-70.,-71.],"Latitude":[-30.,-31.],
                    "Rb_icpms":[3.,4.],"Mg_Number":[.3,.4],"Arc":["a","b"],"Domain":["c","d"]})
    y=pd.Series([5.,6.]);g=pd.Series(["g1","g2"])
    schema=FeatureSchema(list(x.columns[:5]),["Arc","Domain"],2,"stable_v1")
    p=PreparedModel1Data(x,x,"Li_icpms",schema,x,y,PipelineAudit(),g)
    for name,removed in ABLATIONS.items():
        variant=feature_variant(p,name)
        assert variant.y is y and variant.groups is g
        assert variant.x.index.equals(x.index)
        assert list(variant.x)==[c for c in x if c not in removed]
    with pytest.raises(ValueError,match="Unknown"):
        feature_variant(p,"chosen_from_test")
