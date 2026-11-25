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
