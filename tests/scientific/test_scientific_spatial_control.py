import pandas as pd
import pytest
from lithiumscope.experiments.validation.spatial_control import offset,plan,verify_existing_pairs


def test_plan_has_36_positions_for_four_centers():
    centers=pd.DataFrame([{"image_id":name+"_2025","year":2025,"status":"ACCEPT",
        "latitude":-23.1,"longitude":-68.18,"scene_id":name,"path":"/cache.npy"}
        for name in ("atacama","tara","maricunga","pedernales")])
    result=plan(centers)
    assert len(result)==36
    assert result.distance_m.value_counts().to_dict()=={400:16,1000:16,0:4}


def test_east_offset_changes_only_longitude():
    lat,lon=offset(-23.1,-68.18,400,"east")
    assert lat==-23.1
    assert -68.18 < lon < -68.175


def test_identical_scene_required_for_controlled_pairs(tmp_path):
    center=pd.DataFrame([{"image_id":"tara_2025","year":2025,"scene_id":"scene_A",
                          "path":"a","latitude":-23,"longitude":-67}])
    shifted=pd.DataFrame([{"image_id":"tara_east_2025","scene_id":"scene_B",
                           "path":"b","latitude":-23,"longitude":-66.996}])
    result=verify_existing_pairs(center,shifted,tmp_path)
    assert pd.read_csv(result).same_scene.tolist()==[False]
