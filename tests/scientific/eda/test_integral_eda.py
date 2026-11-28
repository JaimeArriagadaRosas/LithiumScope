import json
import pandas as pd

from lithiumscope.model_1.eda.pipeline import run_eda as run_m1
from lithiumscope.model_2.eda.pipeline import run_eda as run_m2
from lithiumscope.results.eda_artifacts import verify_eda


def test_m1_eda_reuses_preparation_and_audits_exclusions(tmp_path):
    data = tmp_path / "samples.csv"
    pd.DataFrame({
        "Li_icpms": [0, 10, -1, 1000, 5],
        "SUM (no water)": [98, 98, 98, 98, 120],
        "SiO2": [51, 52, 53, 54, None],
        "Longitude": [-70, -71, -72, -73, -74],
        "Latitude": [-20, -21, -22, -23, -24],
        "source_dataset": ["Mamani09", "Mamani09", "GEOROC", "GEOROC", "GEOROC"],
    }).to_csv(data, index=False)
    original = data.read_bytes()
    dest = tmp_path / "results"
    result = run_m1(data, output_root=dest)
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows_raw"] == 5
    assert manifest["rows_prepared"] == 3
    assert manifest["scope"] == "descriptive_full_dataset"
    assert manifest["training_reusable"] is False
    profiles = pd.read_csv(result / "tables" / "feature_profiles.csv")
    assert set(profiles.source) >= {"Mamani09", "GEOROC"}
    exclusions = pd.read_csv(result / "tables" / "stage_exclusions.csv")
    assert set(exclusions.stage) >= {"05_target_validity", "06_quality_control"}
    assert verify_eda(result, manifest["signature"])
    assert run_m1(data, output_root=dest) == result
    (result / "tables" / "feature_profiles.csv").write_text("broken")
    assert not verify_eda(result, manifest["signature"])
    assert run_m1(data, output_root=dest) != result
    assert data.read_bytes() == original


def test_m2_eda_preserves_manifest_identity_and_reports_quality(tmp_path, monkeypatch):
    from lithiumscope.model_2.eda import pipeline
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "sample_id": ["a", "b"], "Li_icpms": [10, 20],
        "spatial_group": ["g1", "g2"], "longitude": [-70, -71], "latitude": [-20, -21],
        "sentinel_scene_id": ["s1", "s2"], "sentinel_cloud_cover": [1, 2],
    }).to_csv(manifest_path, index=False)
    config = {"training": {"lithium_column": "Li_icpms", "spatial_group_column": "spatial_group"},
              "model": {"high_lithium_quantile": 0.75}, "imagery": {"bands": ["red"]}}
    monkeypatch.setattr(pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(pipeline, "load_model_2_training_frame", lambda _: pd.DataFrame({
        "red_mean": [0.1, 0.2], "Li_icpms": [10, 20], "spatial_group": ["g1", "g2"],
    }))
    result = run_m2(manifest_path, output_root=tmp_path / "results")
    record = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    assert record["rows_prepared"] == 2
    assert record["features"] == 1
    assert record["training_reusable"] is False
    assert (result / "tables" / "scene_quality.csv").is_file()
    assert verify_eda(result, record["signature"])


def test_qc_review_does_not_admit_rows_or_substitute_missing_oxides(tmp_path):
    from lithiumscope.datasets.adapters.georoc.schema import MAJOR_OXIDES
    frame = pd.DataFrame({name: [9.8, 9.8, 9.8] for name in MAJOR_OXIDES})
    frame["Li_icpms"] = [10, 20, 0]
    frame["SUM (no water)"] = [None, None, 98]
    frame.loc[1, "CaO"] = None
    frame["Longitude"] = [-70, -71, -72]
    frame["Latitude"] = [-20, -21, -22]
    frame["source_dataset"] = ["GEOROC"] * 3
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False)
    original = path.read_bytes()
    result = run_m1(path, output_root=tmp_path / "results")
    review = pd.read_csv(result / "tables/qc_recovery_review.csv")
    assert len(review) == 2
    assert set(review.status) == {"review_sum_in_range", "insufficient_original_measurements"}
    assert not review.approved_for_training.any()
    assert not review.water_basis_and_units_certified.any()
    target = pd.read_csv(result / "tables/target_analytical_review.csv")
    assert "reported_zero" in set(target.status)
    assert path.read_bytes() == original
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["rows_prepared"] == 1


def test_m1_identity_and_buffer_diagnostics_preserve_frozen_folds(tmp_path):
    from lithiumscope.model_1.eda.pipeline import prepare_training_data, materialize_regression_splits
    from lithiumscope.model_1.eda.spatial_review import identity_review, spatial_sensitivity
    path = tmp_path / "data.csv"
    pd.DataFrame({"Li_icpms": [10, 90, 10, 20, 10, 20],
        "SUM (no water)": [98]*6, "Longitude": [-70.001, -70.001, -71., -72., -73., -74.],
        "Latitude": [-20.]*6, "SiO2": [50.]*6,
        "source_sample": ["same", "same", "a", "b", "c", "d"]}).to_csv(path,index=False)
    prepared = prepare_training_data(path)
    identity, pairs = identity_review(prepared)
    assert identity.shared_name.sum() == 2
    assert len(pairs) == 1 and not pairs.target_used.any()
    splits, _ = materialize_regression_splits(prepared.x,prepared.y,prepared.groups,5,42,require_groups=True)
    result = spatial_sensitivity(prepared, splits, identity, pairs)
    sensitivity = result["spatial_buffer_sensitivity"]
    assert len(sensitivity) == 20 and not sensitivity.applied_to_training.any()
    assert sensitivity.groupby("nominal_buffer_m").test_rows_unchanged.sum().eq(6).all()
    assert len(result["outer_chemical_support"]) == 6
