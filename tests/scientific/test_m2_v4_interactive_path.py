"""Regression: normal menu choice 2 trains M2-v4 inputs, never the old v3.

The tests NEVER launch a real optimization, fit, or publish a release.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.training import competition
from lithiumscope.model_2.training.competition_prepare_scientific_data import (
    _prepare_scientific_data,
)
from lithiumscope.model_2.training.v4_inputs import (
    ALGORITHMS,
    combine_v4_quality_mask,
    load_spectral_v4_training_frame,
    spectral_v4_enabled,
    verify_spectral_v4_source,
)


# Historical training matrices, QA inventories and imagery caches are
# deliberately untracked and cannot be present on clean GitHub runners.
# These are explicit OPTIONAL integration checks, not unit-test substitutes.
LOCAL_V4_AVAILABLE = all(
    (PROJECT_ROOT / name).is_file()
    for name in (
        "data/processed/model_2/training_manifest.csv",
        "results/scientific_simulations/m2_rededge_v4_20261113/cohort_706/training_v4_170.csv",
        "results/scientific_simulations/m2_rededge_v4_20261113/cohort_706/rededge_quality.csv",
        "results/scientific_simulations/sentinel_methodology_20261112/inventory/local_patch_inventory_783.csv",
    )
)
LOCAL_V4_EXPANDED_AVAILABLE = LOCAL_V4_AVAILABLE and all(
    (PROJECT_ROOT / name).is_file()
    for name in (
        "data/processed/model_2/vnext_expanded/training_manifest.csv",
        "results/scientific_simulations/m2_rededge_v4_20261113/cohort_783/training_v4_170.csv",
    )
)
requires_local_v4 = pytest.mark.skipif(
    not LOCAL_V4_AVAILABLE,
    reason="Requires local 706-sample real Sentinel-2 matrices not tracked in Git",
)
requires_local_expanded_v4 = pytest.mark.skipif(
    not LOCAL_V4_EXPANDED_AVAILABLE,
    reason="Requires local 783-sample provisional extension not tracked in Git",
)


@pytest.fixture
def model2_v4():
    cfg = load_config("model_2")
    assert spectral_v4_enabled(cfg)
    return cfg


def test_six_algorithms_and_no_new_menu_options(model2_v4):
    from lithiumscope.cli.train_command import _run_impl

    source = Path(__import__("inspect").getfile(_run_impl)).read_text(encoding="utf-8")
    assert tuple(model2_v4["model"]["algorithms"]) == ALGORITHMS
    assert "Seleccione entrenamiento [0-3]" in source
    assert "Entrenar ambos" in source
    assert "spectral_v4" in source
    assert model2_v4["training"]["expansion"]["enabled"] is False


@requires_local_v4
def test_real_v4_source_has_170_and_original_126(model2_v4):
    manifest = PROJECT_ROOT / model2_v4["training"]["manifest_path"]
    matrix, state = verify_spectral_v4_source(manifest, model2_v4)
    assert state["feature_extractor_version"] == 4
    assert state["source_count"] == 706
    frame = load_spectral_v4_training_frame(manifest, model2_v4)
    assert frame.shape == (706, 172)
    assert matrix.is_file()
    assert list(frame.columns[:2]) == ["Li_icpms", "spatial_group"]


@requires_local_v4
def test_quality_mask_excludes_only_train_not_test(model2_v4):
    from lithiumscope.model_2.training.quality import training_quality_mask

    manifest = PROJECT_ROOT / model2_v4["training"]["manifest_path"]
    quality, _ = training_quality_mask(
        manifest,
        inventory_path=PROJECT_ROOT / model2_v4["quality_training"]["inventory_path"],
        minimum_joint_valid=0.95,
    )
    combined, evidence = combine_v4_quality_mask(quality, manifest, model2_v4)
    assert len(combined) == 706
    assert int(combined.sum()) == 690
    assert evidence["all_outer_test_retained"] is True
    assert evidence["original_and_new_spectral_quality"]["all_outer_test_retained"] is True


@requires_local_v4
def test_main_data_preparation_uses_v4_and_spatial_folds_no_fit(
    model2_v4,
    tmp_path,
):
    manifest = PROJECT_ROOT / model2_v4["training"]["manifest_path"]
    context = SimpleNamespace(manifests=tmp_path)
    output = _prepare_scientific_data(
        model2_v4,
        context,
        None,
        model2_v4["training"]["lithium_column"],
        manifest,
        0.75,
        5,
        42,
        "spatial_group",
    )
    _, frame, groups, audit, _, quality, folds, _, x, y, _ = output
    assert frame.shape == (706, 172)
    assert x.shape == (706, 170)
    assert len(groups) == 706 and len(y) == 706
    assert int(quality.sum()) == 690
    assert audit["all_outer_test_retained"] is True
    assert sum(len(f.test_idx) for f in folds) == 706
    assert (tmp_path / "spectral_v4_source.json").is_file()
    assert (tmp_path / "scientific_folds.json").is_file()


@requires_local_v4
def test_first_real_main_run_must_use_v4_matrix_signature(
    model2_v4,
    monkeypatch,
):
    from lithiumscope.model_2.training import competition as c

    manifest = PROJECT_ROOT / model2_v4["training"]["manifest_path"]
    matrix, _ = verify_spectral_v4_source(manifest, model2_v4)
    captured = []

    class PreventExecution(Exception):
        pass

    from lithiumscope.model_2.training import v4_inputs

    original_check = v4_inputs.verify_spectral_v4_source

    def record_verified_matrix(input_manifest, input_cfg):
        source, evidence = original_check(input_manifest, input_cfg)
        captured.append(source.resolve())
        return source, evidence

    monkeypatch.setattr(c, "set_global_seed", lambda *a, **kw: None)
    monkeypatch.setattr(v4_inputs, "verify_spectral_v4_source", record_verified_matrix)
    monkeypatch.setattr(
        c.RunContext,
        "create",
        lambda *a, **kw: (_ for _ in ()).throw(PreventExecution),
    )
    with pytest.raises(PreventExecution):
        c.run_model_2_competition(manifest, SimpleNamespace(accelerator="cpu"))
    assert captured == [matrix.resolve()]


def test_menu_option_two_routes_normal_competition_without_fit(monkeypatch, capsys):
    from lithiumscope.cli import train_command
    from lithiumscope.model_2.training import competition as c

    calls = []
    monkeypatch.setattr(train_command, "choose", lambda *_a, **_kw: "2")
    monkeypatch.setattr(train_command, "require_full_training_environment", lambda: None)
    monkeypatch.setattr(
        train_command,
        "detect_device",
        lambda **kw: SimpleNamespace(accelerator="cpu", name="CPU"),
    )
    monkeypatch.setattr(train_command, "device_summary", lambda *_: None)
    manifest = PROJECT_ROOT / "data/processed/model_2/training_manifest.csv"
    monkeypatch.setattr(train_command, "require_model_2_expanded_manifest", lambda: manifest)

    @contextmanager
    def fake_logger(*_a, **_kw):
        yield "mock"

    monkeypatch.setattr(train_command, "run_log", fake_logger)
    monkeypatch.setattr(train_command, "register_training_run", lambda *a: None)

    def fake_competition(mpath, device):
        calls.append((mpath, device.accelerator))
        return SimpleNamespace(
            run_dir=PROJECT_ROOT / "results/test_only",
            winner=None,
            ranking=pd.DataFrame(),
            failed=[],
        )

    monkeypatch.setattr(c, "run_model_2_competition", fake_competition)
    train_command._run_impl()
    displayed = capsys.readouterr().out
    assert calls == [(manifest, "cpu")]
    assert "Modelo 2 v4" in displayed and "170 características" in displayed
    assert "2. Modelo 2 " in displayed and "Competencia de prospectividad espacial" in displayed
    assert "Modelo 2: RF, Extra Trees, HGB, XGBoost, CatBoost y SVM-RBF." in displayed


@requires_local_expanded_v4
def test_v4_extended_783_can_be_selected_by_config_without_menu_changes(model2_v4):
    from copy import deepcopy

    extended = deepcopy(model2_v4)
    extended["training"]["expansion"]["enabled"] = True
    manifest = PROJECT_ROOT / "data/processed/model_2/vnext_expanded/training_manifest.csv"
    matrix, status = verify_spectral_v4_source(manifest, extended)
    assert matrix.is_file() and status["source_count"] == 783
    frame = load_spectral_v4_training_frame(manifest, extended)
    assert frame.shape == (783, 172)


def test_v4_quality_source_missing_must_fail_closed(model2_v4, tmp_path, monkeypatch):
    from lithiumscope.model_2.training import v4_inputs

    monkeypatch.setattr(v4_inputs, "PROJECT_ROOT", tmp_path)
    fake_manifest = tmp_path / model2_v4["training"]["manifest_path"]
    fake_manifest.parent.mkdir(parents=True)
    fake_manifest.write_text("sample_id,Li_icpms,spatial_group\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="incompletos"):
        v4_inputs.verify_spectral_v4_source(fake_manifest, model2_v4)


def test_v4_refuses_missing_algorithm_without_changing_main_menu(model2_v4):
    from copy import deepcopy
    from lithiumscope.model_2.training.v4_inputs import spectral_v4_paths

    damaged = deepcopy(model2_v4)
    damaged["model"]["algorithms"] = list(ALGORITHMS[:-1])
    manifest = PROJECT_ROOT / damaged["training"]["manifest_path"]
    with pytest.raises(ValueError, match="six"):
        spectral_v4_paths(manifest, damaged)
