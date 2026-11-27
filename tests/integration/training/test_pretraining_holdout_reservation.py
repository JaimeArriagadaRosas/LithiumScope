"""Prospective reservation must precede training and keep labels external."""
import json
import pytest
from lithiumscope.evaluation.holdout.reservation import reserve

def test_reservation_rejects_label_file_in_plan(tmp_path):
    plan=tmp_path/"plan.json"
    plan.write_text(json.dumps({"labels_path":"labels.csv"}))
    with pytest.raises(ValueError,match="labels must remain"):
        reserve(tmp_path/"reservation",plan_path=plan,manifest_path=tmp_path/"manifest.csv",fixed_endpoint=21.6)
    assert not (tmp_path/"reservation").exists()

def test_reservation_cannot_overwrite_existing_root(tmp_path):
    with pytest.raises(FileExistsError):
        reserve(tmp_path,plan_path=tmp_path/"plan.json",manifest_path=tmp_path/"manifest.csv",fixed_endpoint=21.6)


def test_historical_scene_column_is_checked_before_imagery_read():
    import pandas as pd
    from lithiumscope.evaluation.holdout.reservation import _training_overlap
    development=pd.DataFrame({"sample_id":["001"],"spatial_group":["a"],"latitude":[-20.],
        "longitude":[-70.],"image_path":["absent.npy"],"sentinel_scene_id":["shared"]})
    external=pd.DataFrame({"sample_id":["002"],"spatial_group":["b"],"latitude":[-30.],
        "longitude":[-70.],"scene_id":["shared"],"image_sha256":["x"]})
    with pytest.raises(ValueError,match="scene_id"):
        _training_overlap(external,development,1000)
