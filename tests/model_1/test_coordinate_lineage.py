from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lithiumscope.core.pipeline_audit import PipelineAudit, stable_source_row_id
from lithiumscope.datasets.harmonization import canonicalize_coordinates
from lithiumscope.datasets.validation.coordinates import assess_coordinate_aliases, coordinate_summary
from lithiumscope.model_1.validation.coordinate_preflight import coordinate_preflight
from lithiumscope.model_1.validation.spatial import build_spatial_groups
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.model_1.prediction.predictor import predict_model_1_frame


def test_complementary_conflicting_missing_invalid_and_reported_crs():
    raw=pd.DataFrame({
        "Longitude":[-70.,-70.,None,170.,-70.,180.,-70.,-70.,None],
        "Longitude (X)":[None,-71.,-69.,None,None,-180.,-69.999994,-70.,None],
        "Logintude (X)":[None,None,None,None,None,None,-70.000006,None,None],
        "Latitude":[-20.,-20.,-20.,-100.,-20.,20.,-20.,-20.,None],
        "Latitude (Y)":[-20.,None,None,None,100.,None,None,None,None],
        "source_coordinate_crs":["unknown","unknown","EPSG:4326","unknown","unknown","WGS84","unknown","EPSG:32719","unknown"],
        "coordinate_precision":[None,1.,None,None,None,None,None,20.,None],
    })
    # Row 3's latitude is outside range; no silent swapping.
    checked,evidence=assess_coordinate_aliases(raw)
    assert evidence.coordinate_status.tolist()==[
        "valid_numeric_coordinates","conflicting_aliases","valid_numeric_coordinates","invalid_alias",
        "possible_axis_swap","valid_numeric_coordinates","conflicting_aliases","requires_crs_review","missing_coordinate"]
    assert checked.loc[2,"Longitude"]==-69 and checked.loc[5,"Longitude"]==-180
    assert checked.loc[[1,3,4,6,7,8],["Latitude","Longitude"]].isna().all().all()
    assert evidence.loc[1,"original_Longitude"]==-70
    assert evidence.loc[7,"original_coordinate_precision"]==20
    assert coordinate_summary(evidence)["unknown_crs_rows"]==6
    assert not coordinate_summary(evidence)["coordinate_accuracy_certified"]
    assert build_spatial_groups(checked) is None
    assert canonicalize_coordinates(raw).loc[1,"Longitude"]==-70
    assert pd.isna(canonicalize_coordinates(raw,strict=True).loc[1,"Longitude"])


def test_canonical_coordinate_precedes_case_variant():
    f=pd.DataFrame({"Longitude":[-70.],"longitude":[-120.],"Latitude":[-20.],"latitude":[0.]})
    assert build_spatial_groups(f).iloc[0]=="g_140_220"
    result,evidence=assess_coordinate_aliases(f)
    assert evidence.coordinate_status.iloc[0]=="conflicting_aliases"
    assert pd.isna(result.Longitude.iloc[0])


def test_spatial_preflight_blocks_conflict_and_declares_unknown_crs(tmp_path):
    p=SimpleNamespace(coordinate_report={"unresolved_coordinate_rows":1,"unknown_crs_rows":2},
                      groups=None,coordinate_evidence=pd.DataFrame({"coordinate_status":["conflicting_aliases"]}))
    _,_,errors,warnings=coordinate_preflight(p,SimpleNamespace(tables=tmp_path),require_groups=True,outer_folds=5)
    assert any("unresolved" in text for text in errors)
    assert any("datum" in text for text in warnings)
    assert (tmp_path/"coordinate_alias_audit.csv").is_file()


def test_prediction_marks_conflicting_coordinates_non_applicable():
    raw=pd.DataFrame({"Longitude":[-70.],"Longitude (X)":[-80.],"Latitude":[-20.],"Na2O":[3.]})
    schema=FeatureSchema(["Longitude","Latitude","Na2O"],[],2)
    class Estimator:
        def predict(self,x):
            assert pd.isna(x.Longitude.iloc[0]) and pd.isna(x.Latitude.iloc[0])
            return np.array([10.])
    bundle={"schema":schema,"algorithm":"dummy","estimator":Estimator(),
            "applicability_profile":{c:{"q01":-180.,"q99":180.} for c in schema.numeric}}
    result,diag=predict_model_1_frame(raw,bundle=bundle)
    assert result.applicability_warning.iloc[0]=="INSUFFICIENT_COORDINATE_EVIDENCE"
    assert diag["coordinate_report"]["conflicting_alias_rows"]==1


def test_stage_lineage_rejects_unexplained_filtering_and_identity_changes():
    frame=pd.DataFrame({"source_sample":["a","b","c"],"x":[1.,2.,3.]},index=[10,20,30])
    audit=PipelineAudit(dataset_sha256="a"*64)
    audit.capture("initial",frame)
    with pytest.raises(ValueError,match="exclusions"):
        audit.capture("filter",frame.iloc[:2])
    with pytest.raises(ValueError,match="reordered"):
        audit.capture("reordered",frame.iloc[::-1])
    with pytest.raises(ValueError,match="changed source"):
        audit.capture("changed",frame.assign(source_sample=["a","b","d"]))
    with pytest.raises(ValueError,match="column change"):
        audit.capture("extra",frame.assign(extra=1))
    audit.capture("filter",frame.iloc[:2],exclusion_reasons=[{"row_index":"30","reason":"invalid_x"}])
    assert audit.exclusions[0]["source_row_id"]==stable_source_row_id("a"*64,30)
    assert audit.exclusions[0]["reason"]=="invalid_x"
    assert audit.to_frame().rows_excluded.tolist()==[0,1]
    repeat=PipelineAudit(dataset_sha256="a"*64)
    repeat.capture("initial",frame.copy())
    assert repeat.records[0]["frame_digest_sha256"]==audit.records[0]["frame_digest_sha256"]
    assert len(audit.column_records)==4
