"""M2 v4 experimental safety guards (no real training or downloads)."""

import pytest

def test_overfit_audit_requires_completed_all_six(tmp_path):
    from scripts.scientific.m2_v4_nested_overfit_audit import audit

    with pytest.raises(FileNotFoundError):
        audit(root=tmp_path)


def test_m1_plus_previous_m2_integration_model_refs_are_distinct():
    from scripts.scientific.m1_new_m2_old_integration import M1, M2

    assert "training_20261106_111653" in str(M1)
    assert "training_20260920_195933" in str(M2)
    assert M1 != M2


def test_paired_77_ablation_requires_both_real_completed_runs(tmp_path):
    from scripts.scientific.m2_v4_paired_77_ablation import compare

    with pytest.raises(FileNotFoundError):
        compare(root=tmp_path)


def test_spectral_identity_auditor_never_guesses_missing_data(tmp_path):
    from scripts.scientific.m2_v4_spectral_identity_audit import analyze

    with pytest.raises(ValueError):
        analyze(100, root=tmp_path)
    with pytest.raises(FileNotFoundError):
        analyze(706, root=tmp_path)


def test_v4_candidate_bundle_requires_six_finished_algorithms(tmp_path):
    from scripts.scientific.m2_v4_bundle_candidate import train_final

    with pytest.raises(FileNotFoundError):
        train_final(root=tmp_path)


def test_v4_inference_rejects_old_six_feature_model(tmp_path):
    from scripts.scientific.m2_v4_inference import score_candidate
    import joblib

    old_bundle = tmp_path / "bad.joblib"
    joblib.dump(
        {
            "algorithm": "catboost",
            "feature_extractor_version": 3,
            "experimental_only": True,
            "features": [f"f{i}" for i in range(126)],
        },
        old_bundle,
    )
    with pytest.raises(ValueError, match="170-feature"):
        score_candidate(
            tmp_path / "nonexistent.npy",
            "S2A_X",
            -23.0,
            -67.0,
            candidate_path=old_bundle,
            spectral_cache=tmp_path,
        )
