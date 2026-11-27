import pandas as pd
import pytest

from lithiumscope.core.states import DatasetState
from lithiumscope.datasets.management.status import DatasetStatus
from lithiumscope.datasets.management.sources import diagnostic, local
from lithiumscope.datasets.management.sources.model_2 import require_frozen_manifest
from lithiumscope.runtime import dataset_provisioning as runtime


def test_configured_preboot_uses_combined_and_expanded_not_old_base(tmp_path, monkeypatch):
    combined = tmp_path / "combined.csv"
    combined.write_text("Li_icpms\n10\n")
    expanded = tmp_path / "expanded.csv"
    pd.DataFrame({"Li_icpms": [10], "spatial_group": ["g"], "sample_id": ["s"],
                  "image_path": ["a.npy"]}).to_csv(expanded, index=False)
    configs = {"model_1": {"data_sources": {"georoc": {"enabled": True, "combined_path": "combined.csv"}}},
               "model_2": {"training": {"expansion": {"enabled": True, "output_manifest_path": "expanded.csv"}}}}
    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(runtime, "load_config", configs.__getitem__)
    monkeypatch.setattr(runtime, "ensure_dataset", lambda *a, **k: pytest.fail("read-only"))
    assert [s.path for s in runtime.inspect_required_datasets()] == [str(combined), str(expanded)]
    expanded.unlink()
    assert runtime.inspect_required_datasets()[1].effective_state == DatasetState.MISSING


def test_diagnostic_acquisition_is_injected_and_rechecks_configured_source(tmp_path, monkeypatch):
    from lithiumscope.core import config
    source = tmp_path / "input.csv"
    def resolve(group):
        if not source.exists():
            raise FileNotFoundError("missing")
        return source
    monkeypatch.setattr(local, "local_training_source", resolve)
    monkeypatch.setattr(config, "load_config", lambda _: {"data_sources": {"georoc": {}}})
    def acquire(group):
        assert group == "model_1"
        source.write_text("Li_icpms\n10\n")
    assert diagnostic.prepare_diagnostic_source("model_1", acquire=True, acquire_source=acquire) == source


def test_image_verification_is_explicit_and_reports_missing(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"Li_icpms": [10], "spatial_group": ["g"], "sample_id": ["001"],
                  "image_path": ["a.npy"]}).to_csv(manifest, index=False)
    training = {"manifest_path": "manifest.csv", "frozen_expected_rows": 1}
    assert require_frozen_manifest(training, tmp_path) == manifest
    with pytest.raises(ValueError, match="image references"):
        require_frozen_manifest(training, tmp_path, verify_images=True)
    (tmp_path / "a.npy").write_bytes(b"fixture")
    assert require_frozen_manifest(training, tmp_path, verify_images=True) == manifest


def test_dataset_state_rejects_contradiction():
    with pytest.raises(ValueError, match="conflicts"):
        DatasetStatus("x", True, "a.csv", state=DatasetState.MISSING)
