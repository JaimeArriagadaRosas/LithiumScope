from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.shared.curve_suite import summarize_curves, run_curve_suite
from lithiumscope.resources.detection.device import DeviceInfo


def test_summary_averages_seeds_inside_folds_and_preserves_failures():
    table = pd.DataFrame({"algorithm": ["a"]*4, "fraction_of_eligible_groups": [.5]*4,
                          "fold": [1, 1, 2, 2], "status": ["ok", "ok", "ok", "failed"],
                          "test_rmse": [1., 3., 8., np.nan]})
    row = summarize_curves(table).iloc[0]
    assert row.test_rmse_mean == 5.
    assert row.state == "incomplete"
    assert row.failed_partitions == 1
    assert row.test_rmse_valid_folds == 2


def test_preflight_never_fits_and_persists_partition_identity(tmp_path, monkeypatch):
    from lithiumscope.experiments.shared import curve_suite
    from sklearn.model_selection import GroupKFold
    groups = pd.Series(np.repeat(np.arange(6), 4))
    y = pd.Series(np.arange(24, dtype=float))
    x = pd.DataFrame({"x": y})
    splits = list(GroupKFold(3).split(x, y, groups))
    monkeypatch.setattr(curve_suite, "evaluation_identity", lambda *a, **kw: "verified-test")
    monkeypatch.setattr(curve_suite, "runtime_fingerprint", lambda: {})
    device = DeviceInfo("cpu", "test", 1, False, False, False)
    result = run_curve_suite(x=x, y=y, groups=groups, splits=splits, kind="regression",
        algorithms=["dummy"], device=device, policy={}, output=tmp_path / "preflight",
        fit=lambda *a: pytest.fail("preflight called fit"), predict=None,
        fractions=(.5, 1.), seeds=(42,))
    assert result["state"] == "preflight"
    assert not result["fitted_models_saved"]
    assert len(result["partitions"]) == 6
    assert (tmp_path / "preflight" / "curve_manifest.json").is_file()


def test_execution_keeps_results_even_when_one_family_fails(tmp_path, monkeypatch):
    from lithiumscope.experiments.shared import curve_suite
    from sklearn.model_selection import GroupKFold
    from sklearn.dummy import DummyRegressor
    groups = pd.Series(np.repeat(np.arange(6), 4))
    y = pd.Series(np.arange(24, dtype=float))
    x = pd.DataFrame({"x": y})
    splits = list(GroupKFold(3).split(x, y, groups))
    monkeypatch.setattr(curve_suite, "evaluation_identity", lambda *a, **kw: "verified-test")
    monkeypatch.setattr(curve_suite, "runtime_fingerprint", lambda: {})
    def fit(algorithm, fold, a, b, g):
        if algorithm == "failed":
            raise RuntimeError("synthetic failure")
        return DummyRegressor().fit(a, b)
    out = tmp_path / "run"
    result = run_curve_suite(x=x, y=y, groups=groups, splits=splits, kind="regression",
        algorithms=["dummy", "failed"], device=DeviceInfo("cpu", "test", 1, False, False, False),
        policy={}, output=out, fit=fit, predict=lambda model, a: model.predict(a), execute=True,
        fractions=(1.,), seeds=(42,))
    assert result["state"] == "incomplete"
    table = pd.read_csv(out / "curve_partitions.csv")
    assert len(table) == 6
    assert set(table.status) == {"ok", "failed"}
    assert not list(out.glob("*.joblib"))
