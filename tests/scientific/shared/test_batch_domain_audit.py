import pandas as pd
from lithiumscope.experiments.generalization.batch_audit import summarize_batches
from lithiumscope.experiments.generalization.domain import predictor_support


def test_batch_source_mapping_and_target_range(tmp_path):
    source=tmp_path/"data.csv"
    pd.DataFrame({"Sample_material_batch_id":[3,3,4],"Li_icpms":[10,20,30]}).to_csv(source,index=False)
    result=summarize_batches(source,{"row_ids":["0","2"],"target":[10.,30.],"groups":["a","b"]})
    assert result["numeric_target_rows_verified"]==2
    assert result["batches"][0]["maximum"]==30
    assert result["batches"][1]["raw_rows"]==2


def test_predictor_support_uses_train_ranges_only():
    x=pd.DataFrame({"a":[1.,2.],"b":[None,None],"cat":["A","B"]})
    y=pd.DataFrame({"a":[3.,1.5],"b":[10.,None],"cat":["C","A"]})
    result=predictor_support(x,y,["a","b"],["cat"])
    assert result["outside_numeric_range"].tolist()==[1,0]
    assert result["unsupported_numeric_features"].tolist()==[1,0]
    assert result["unseen_categories"].tolist()==[1,0]
