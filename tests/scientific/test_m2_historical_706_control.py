"""M2 706-only source control. Tests neither fit models nor alter datasets."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from lithiumscope.datasets import provisioner
from lithiumscope.model_2.training import competition
from lithiumscope.model_2.training.diagnostic_inputs import loaded_competitions
from lithiumscope.model_2.training.postprocess import load_predictions
from lithiumscope.model_2.training.quality import training_quality_mask


def test_disabled_expansion_routes_only_to_frozen_manifest(monkeypatch, tmp_path):
    historic = tmp_path / "frozen_706.csv"
    historic.write_text("sample_id\ns1\n", encoding="utf-8")
    monkeypatch.setattr(
        provisioner, "load_config",
        lambda _: {"training": {"expansion": {"enabled": False}}},
    )
    monkeypatch.setattr(
        provisioner, "require_frozen_model_2_manifest", lambda: historic,
    )
    assert provisioner.require_model_2_expanded_manifest() == historic


def test_competition_rejects_expanded_source_in_historical_mode(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(competition, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        competition, "load_config",
        lambda _: {"training": {
            "manifest_path": "frozen_706.csv",
            "expansion": {"enabled": False},
        }},
    )
    with pytest.raises(ValueError, match="frozen 706-row"):
        competition.run_model_2_competition(tmp_path / "expanded_783.csv", None)


def test_historical_qa_uses_verified_prefix_without_modifying_inventory(tmp_path):
    image = tmp_path / "reference.npy"
    np.save(image, np.ones((6, 32, 32), dtype="float32"))
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    sample_ids = [f"S{i:04d}" for i in range(783)]
    manifest = tmp_path / "historical.csv"
    pd.DataFrame({
        "sample_id": sample_ids[:706], "image_path": [str(image)] * 706,
        "Li_icpms": np.resize([10.0, 40.0], 706),
        "spatial_group": [f"G{i // 5}" for i in range(706)],
    }).to_csv(manifest, index=False)
    inventory = tmp_path / "inventory_783.csv"
    quality = [1.0] * 783
    quality[2] = .8
    quality[705] = .6
    pd.DataFrame({
        "manifest_index": range(783),
        "sample_id": sample_ids,
        "image_sha256": [digest] * 783,
        "sixband_joint_valid_fraction": quality,
    }).to_csv(inventory, index=False)
    original_bytes = inventory.read_bytes()
    mask, summary = training_quality_mask(
        manifest, inventory_path=inventory,
        minimum_joint_valid=.95,
    )
    assert len(mask) == 706 and int(mask.sum()) == 704
    assert summary["training_excluded"] == 2
    assert summary["all_test_rows_preserved"] is True
    assert inventory.read_bytes() == original_bytes
    tampered = pd.read_csv(inventory)
    tampered.loc[705, "sample_id"] = "wrong"
    tampered.to_csv(inventory, index=False)
    with pytest.raises(ValueError, match="order differs"):
        training_quality_mask(manifest, inventory_path=inventory)


def _save_oof(folder, lithium, *, count):
    folder.mkdir(parents=True, exist_ok=True)
    label = np.arange(count) % 4 == 0
    for name in ("random_forest", "extra_trees"):
        pd.DataFrame({
            "y_true": label.astype(int),
            "li_ppm": lithium[:count],
            "outer_fold": np.arange(count) % 5 + 1,
            "prospectivity_score": np.where(label, .7, .2),
        }).to_csv(folder / f"oof_predictions_{name}.csv", index=False)


def test_oof_loader_accepts_706_and_rejects_wrong_size(tmp_path):
    li = np.resize([5.0, 20.0], 783)
    df = pd.DataFrame({"Li_icpms": li[:706]})
    tables = tmp_path / "tables"
    _save_oof(tables, li, count=706)
    scores, folds, target = load_predictions(df, tables)
    assert set(scores) == {"random_forest", "extra_trees"}
    assert len(folds) == 706 and len(target) == 706
    pd.DataFrame({
        "y_true": np.zeros(783), "li_ppm": li,
        "outer_fold": np.arange(783) % 5 + 1,
        "prospectivity_score": np.full(783, .2),
    }).to_csv(tables / "oof_predictions_random_forest.csv", index=False)
    with pytest.raises(ValueError, match="Incomplete OOF"):
        load_predictions(df, tables)


def test_ablation_compares_same_706_test_rows_against_previous_783(tmp_path):
    root = tmp_path
    historical = root / "historical_706.csv"
    expanded = root / "data/processed/model_2/vnext_expanded/training_manifest.csv"
    expanded.parent.mkdir(parents=True)
    ids = [f"S{i:04d}" for i in range(783)]
    li = np.resize([5.0, 20.0], 783)
    cohort = pd.DataFrame({
        "sample_id": ids, "Li_icpms": li,
        "spatial_group": [f"G{i // 5}" for i in range(783)],
    })
    cohort.to_csv(expanded, index=False)
    cohort.iloc[:706].to_csv(historical, index=False)
    current_tables = root / "current"
    previous_tables = root / "previous"
    _save_oof(current_tables, li, count=706)
    _save_oof(previous_tables, li, count=783)
    frame, current, previous, audit = loaded_competitions(
        historical, current_tables, root=root, older_tables=previous_tables,
    )
    assert len(frame) == 706
    assert len(current["random_forest"]) == 706
    assert len(previous["extra_trees"]) == 706
    assert audit["metadata"]["historical_706_vs_expanded_783_ablation"]
    assert audit["metadata"]["previous_training_source_rows"] == 783
    assert audit["metadata"]["same_folds_and_labels"]


def test_ablation_rejects_mutated_historical_prefix(tmp_path):
    historic = tmp_path / "h.csv"
    old = tmp_path / "data/processed/model_2/vnext_expanded/training_manifest.csv"
    old.parent.mkdir(parents=True)
    cohort = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(783)],
        "Li_icpms": np.resize([5.0, 20.0], 783),
        "spatial_group": [f"G{i // 3}" for i in range(783)],
    })
    cohort.to_csv(old, index=False)
    historical = cohort.iloc[:706].copy()
    historical.loc[0, "sample_id"] = "tampered"
    historical.to_csv(historic, index=False)
    current = tmp_path / "current"
    previous = tmp_path / "previous"
    _save_oof(current, historical.Li_icpms.to_numpy(float), count=706)
    _save_oof(previous, cohort.Li_icpms.to_numpy(float), count=783)
    with pytest.raises(ValueError, match="preserve frozen 706"):
        loaded_competitions(
            historic, current, root=tmp_path, older_tables=previous,
        )
