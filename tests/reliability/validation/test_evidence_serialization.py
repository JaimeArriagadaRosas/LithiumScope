import json

import numpy as np
import pandas as pd

from lithiumscope.tools.inspection.development import snapshot


def test_undefined_scientific_values_are_null_not_zero():
    result = snapshot._section("nested", lambda: {"state": "OK", "rows": [
        {"distance": np.float64(3.5), "support": np.int64(2), "valid": np.bool_(True)},
        {"distance": float("nan"), "support": pd.NA, "limit": float("inf")},
    ]})
    assert result["rows"][1] == {"distance": None, "support": None, "limit": None}
    assert result["undefined_numeric_values"] == 3
    assert json.loads(json.dumps(result, allow_nan=False))["rows"][0]["support"] == 2


def test_unsupported_section_does_not_break_other_sections():
    bad = snapshot._section("bad", lambda: {"state": "OK", "value": object()})
    good = snapshot._section("good", lambda: {"state": "OK", "count": 7})
    assert bad["state"] == "ERROR"
    assert good["count"] == 7
    json.dumps({"bad": bad, "good": good}, allow_nan=False)


def test_snapshot_publishes_only_complete_json(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "collect_snapshot", lambda root: {"state": "PARTIAL"})
    report = snapshot.write_snapshot(tmp_path)
    assert json.loads(report.read_text())["state"] == "PARTIAL"
    assert list(report.parent.iterdir()) == [report]
