"""Missing-source authorization and immutable snapshot policy."""
import pytest
from lithiumscope.datasets.management.diagnostic_provisioning import prepare_diagnostic_source


def test_no_acquisition_by_default(monkeypatch):
    from lithiumscope.datasets.management import local_sources
    def missing(_): raise FileNotFoundError("missing")
    monkeypatch.setattr(local_sources,"local_training_source",missing)
    with pytest.raises(FileNotFoundError,match="missing"):
        prepare_diagnostic_source("model_1")


def test_frozen_snapshot_not_reconstructed_even_with_authorization(monkeypatch):
    from lithiumscope.datasets.management import local_sources
    from lithiumscope.core import config
    def missing(_): raise FileNotFoundError("missing")
    monkeypatch.setattr(local_sources,"local_training_source",missing)
    monkeypatch.setattr(config,"load_config",lambda _:{"data_sources":{"georoc":{"enabled":True,"frozen_combined_sha256":"x"}}})
    with pytest.raises(FileNotFoundError,match="Frozen M1"):
        prepare_diagnostic_source("model_1",acquire=True)
