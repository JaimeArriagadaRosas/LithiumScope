from lithiumscope.datasets.status import DatasetStatus
from lithiumscope.runtime.preboot import PrebootReport


def _report(datasets):
    return PrebootReport(
        python_ok=True,
        python_version="3.12.0",
        platform="test",
        core=[],
        ml=[],
        imagery=[],
        dev=[],
        datasets=datasets,
        configs_ok=True,
        writable_ok=True,
        placeholders_removed=0,
        accelerator="cpu",
        accelerator_name="test",
    )


def test_preboot_requires_both_model_datasets():
    report = _report(
        [
            DatasetStatus("model_1_geochemistry", True, "model1.csv", "ok"),
        ]
    )
    assert report.datasets_ready is False


def test_preboot_accepts_both_model_datasets():
    report = _report(
        [
            DatasetStatus("model_1_geochemistry", True, "model1.csv", "ok"),
            DatasetStatus("model_2_sentinel2", True, "manifest.csv", "ok"),
        ]
    )
    assert report.datasets_ready is True
