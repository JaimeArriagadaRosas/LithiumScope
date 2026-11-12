import json
import numpy as np
from lithiumscope.core.training_diagnostics import save_oof_diagnostics


def test_regression_diagnostics_are_grouped_and_reproducible(tmp_path):
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    y = [10, 20, 30, 40, 50, 60]
    pred = [11, 19, 32, 39, 49, 58]
    groups = ["a", "a", "b", "b", "c", "c"]
    one = save_oof_diagnostics(a, y, pred, groups=groups, kind="regression")
    two = save_oof_diagnostics(b, y, pred, groups=groups, kind="regression")
    assert one["metrics"] == two["metrics"]
    assert one["cluster_bootstrap"] == two["cluster_bootstrap"]
    assert one["n_groups"] == 3
    assert one["external_validation"] == "NOT_EVALUATED"


def test_classification_diagnostics_do_not_claim_external_validation(tmp_path):
    output = save_oof_diagnostics(
        tmp_path, [0, 1, 0, 1, 0, 1],
        [.1, .6, .2, .7, .3, .8],
        groups=["a", "b", "c", "d", "e", "f"],
        kind="classification", seed=10,
    )
    assert "roc_auc" in output["metrics"]
    assert json.loads(
        (tmp_path / "scientific_diagnostics.json").read_text(encoding="utf8")
    )["external_validation"] == "NOT_EVALUATED"
