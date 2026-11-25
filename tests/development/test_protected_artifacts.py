from lithiumscope.tools.inspection.development.protected import protected_inventory, compare_protected


def test_protected_inventory_detects_modification_and_never_restores(tmp_path):
    path = tmp_path / "data/processed/model_1/training_combined.csv"
    path.parent.mkdir(parents=True)
    path.write_text("original")
    before = protected_inventory(tmp_path)
    path.write_text("changed")
    result = compare_protected(before, protected_inventory(tmp_path))
    assert result["state"] == "ERROR"
    assert result["changed_paths"] == ["data/processed/model_1/training_combined.csv"]
    assert path.read_text() == "changed"
    assert not result["restoration_attempted"]


def test_protected_inventory_detects_new_model(tmp_path):
    before = protected_inventory(tmp_path)
    model = tmp_path / "models/model_1/trained/run/model.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"not a serialized estimator")
    result = compare_protected(before, protected_inventory(tmp_path))
    assert result["state"] == "ERROR"
    assert "models/model_1/trained/run/model.joblib" in result["changed_paths"]


def test_unchanged_missing_inputs_are_not_modified(tmp_path):
    before = protected_inventory(tmp_path)
    assert compare_protected(before, protected_inventory(tmp_path))["state"] == "OK"
    assert compare_protected({"x": {"state": "BLOCKED"}}, {"x": {"state": "BLOCKED"}})["state"] == "WARNING"
