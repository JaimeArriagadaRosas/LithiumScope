from pathlib import Path
import json

import pytest

from lithiumscope.core.execution import tracker


def test_failed_replace_leaves_previous_manifest_and_no_temp_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "runtime_fingerprint", lambda: {})
    instance = tracker.ExperimentTracker(tmp_path, "run", "model_1")
    before = instance.path.read_bytes()
    def fail_replace(self, target):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        instance.attach_summary(value=7)
    assert instance.path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]
    assert json.loads(before)["summary"] == {}
