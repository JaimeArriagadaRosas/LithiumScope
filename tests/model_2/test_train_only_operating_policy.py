import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.training.calibration.operating import learn_operating_policy, operating_evidence


def test_inner_fit_uses_only_development_and_frozen_cut():
    x = pd.DataFrame({"id": np.arange(24), "signal": np.tile([.1, .8], 12)})
    y = np.tile([0, 1], 12)
    groups = pd.Series(np.repeat(np.arange(12), 2))
    fits = []
    class Model:
        def predict_proba(self, validation):
            assert not set(validation.id) & fits[-1]
            p = validation.signal.to_numpy()
            return np.column_stack([1-p, p])
    def fit(train, labels, groups):
        fits.append(set(train.id))
        assert not fits[-1] & {24, 25}
        assert len(labels) == len(groups) == len(train)
        return Model()
    frozen, table = learn_operating_policy(x, y, groups, fit=fit, row_indices=np.arange(24),
                                           reserved_indices=[24, 25], n_splits=3)
    assert len(fits) == 3
    assert frozen.inner_count == 24
    assert table.row_index.tolist() == list(range(24))
    a = operating_evidence(frozen, table, [0, 1], [.2, .9])
    b = operating_evidence(frozen, table, [1, 0], [.2, .9])
    assert a["operating_threshold"] == b["operating_threshold"]
    assert a["operating_outer_metrics"]["balanced_accuracy"] != b["operating_outer_metrics"]["balanced_accuracy"]


def test_monoclass_inner_split_fails_before_fitting():
    x = pd.DataFrame({"id": range(8)})
    with pytest.raises(ValueError, match="lacks both classes"):
        learn_operating_policy(x, [0,0,0,0,1,1,1,1], [0,0,0,0,1,1,1,1],
            fit=lambda *a: pytest.fail("Must validate first"), row_indices=np.arange(8),
            reserved_indices=[], n_splits=2)

def test_cv_runner_persists_and_reconstructs_train_only_policy(monkeypatch, tmp_path):
    from lithiumscope.model_2.training.execution import cv_runner
    from lithiumscope.model_2.validation.spatial import make_scientific_folds, scientific_labels

    n = 72
    x = pd.DataFrame({"id": np.arange(n), "signal": np.tile([.15, .7, .2, .8], n//4)})
    li = np.tile([5., 30., 10., 40.], n//4)
    groups = pd.Series(np.repeat(np.arange(18), 4))
    folds = make_scientific_folds(pd.Series(li), groups, n_splits=3, fixed_threshold_ppm=20.)
    y = pd.Series(scientific_labels(folds, n))
    fits = []
    class Model:
        def __init__(self, ids):
            self.ids = set(ids)
        def predict_proba(self, data):
            assert set(data.id) == self.ids or not self.ids & set(data.id)
            p = data.signal.to_numpy()
            return np.column_stack([1-p, p])
    def fake_fit(algorithm, device, seed, params, x_train, y_train, groups_train, inner_folds, **kw):
        fits.append(set(x_train.id))
        return Model(x_train.id)
    monkeypatch.setattr(cv_runner, "_fit_estimator", fake_fit)
    args = dict(x=x, y=y, algorithm="catboost", device=None, seed=42, folds=3,
                groups=groups, checkpoint_root=tmp_path, inner_folds=3,
                optimization={"enabled": False, "operating_threshold_objective": "balanced_accuracy"},
                scientific_folds=folds, lithium_values=li, fixed_threshold_ppm=20.)
    first = cv_runner.run_classification_cv(**args)
    assert len(fits) == 12  # 3 internal fits + 1 outer fit per fold
    assert first.fold_table.operating_test_balanced_accuracy.tolist() == [1.] * 3
    assert len(first.inner_operating_predictions) == 144
    for number, fold in enumerate(folds, 1):
        inner = first.inner_operating_predictions.query("outer_fold == @number")
        assert set(inner.row_index) == set(fold.train_idx)
        assert not set(inner.row_index) & set(fold.test_idx)
    second = cv_runner.run_classification_cv(**args)
    assert len(fits) == 12
    assert second.fold_table.resumed.all()
    pd.testing.assert_frame_equal(first.inner_operating_predictions, second.inner_operating_predictions)
    assert first.fold_table.operating_threshold.tolist() == second.fold_table.operating_threshold.tolist()


def test_fast_threshold_matches_brute_force_objective():
    from lithiumscope.model_2.evaluation.thresholds import select_operating_threshold, _candidate_thresholds
    from lithiumscope.core.scientific.classification_metrics import classification_metrics
    rng = np.random.default_rng(42)
    y = np.tile([0, 1], 15)
    p = rng.choice(np.linspace(0, 1, 11), len(y))
    for objective in ("balanced_accuracy", "recall", "precision", "f1", "accuracy"):
        selected = select_operating_threshold(y, p, objective=objective)
        reference = max(classification_metrics(y, p, float(t))[objective] for t in _candidate_thresholds(p))
        assert selected.objective_value == pytest.approx(reference)

def test_prior_baseline_uses_same_qa_fit_rows_and_keeps_full_test():
    from lithiumscope.model_2.validation.spatial import make_scientific_folds, evaluate_scientific_baseline
    li = pd.Series(np.tile([5., 30., 10., 40.], 18))
    groups = pd.Series(np.repeat(np.arange(18), 4))
    folds = make_scientific_folds(li, groups, n_splits=3, fixed_threshold_ppm=20.)
    mask = np.ones(len(li), dtype=bool)
    mask[1::8] = False
    row, table, scores = evaluate_scientific_baseline(folds, len(li), quality_mask=mask)
    assert len(scores) == len(li) and np.isfinite(scores).all()
    for i, f in enumerate(folds):
        expected = f.y_train[mask[f.train_idx]].mean()
        assert np.allclose(scores[f.test_idx], expected)
        assert table.n_train.iloc[i] == mask[f.train_idx].sum()
    assert row["valid_folds"] == 3 and row["n_oof"] == len(li)
