"""Exercise chart content contracts, partial data and figure cleanup."""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from lithiumscope.reporting.figures.charts import save_integrated_figures, _scatter, _case_bar
from lithiumscope.reporting.figures.satellite import _rgb_preview, save_satellite_input_preview


def test_charts_render_all_expected_artifacts(tmp_path):
    data = pd.DataFrame({
        "case_id": ["a", "b", "c"],
        "Li_icpms": [10., 20., 30.], "Li_icpms_predicted": [11., 18., 32.],
        "prospectivity_score": [.2, .6, .9],
        "model_1_out_of_training_range_fraction": [.1, .2, .3],
        "model_2_out_of_training_range_fraction": [.2, .3, .1],
    })
    before = data.copy(deep=True)
    figures = set(plt.get_fignums())
    outputs = save_integrated_figures(data, tmp_path)
    assert {p.name for p in outputs} == {
        "li_real_vs_model_1.png", "model_1_absolute_error_by_case.png",
        "li_real_vs_model_2_score.png", "model_1_vs_model_2_score.png",
        "model_2_score_by_case.png", "model_1_ood_by_case.png",
        "model_2_ood_by_case.png", "model_1_vs_model_2_ood.png",
    }
    for path in outputs:
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width > 0 and image.height > 0
            image.verify()
    pd.testing.assert_frame_equal(data, before)
    assert set(plt.get_fignums()) == figures


def test_charts_skip_missing_nonfinite_and_insufficient_data(tmp_path):
    assert save_integrated_figures(pd.DataFrame(), tmp_path) == []
    frame = pd.DataFrame({
        "case_id": ["a", "b", "c"], "Li_icpms": [10., np.inf, "bad"],
        "Li_icpms_predicted": [11., 20., np.nan],
    })
    paths = save_integrated_figures(frame, tmp_path)
    assert [path.name for path in paths] == ["model_1_absolute_error_by_case.png"]
    assert _case_bar(pd.DataFrame({"case_id": ["a"], "value": [np.inf]}),
                     value_column="value", title="value", ylabel="value",
                     destination=tmp_path / "invalid.png") is None


@pytest.mark.parametrize("kind", ["scatter", "bar"])
def test_save_failure_releases_matplotlib_figure(tmp_path, monkeypatch, kind):
    original = set(plt.get_fignums())
    def fail(*args, **kwargs):
        raise OSError("disk failure")
    monkeypatch.setattr(Figure, "savefig", fail)
    data = pd.DataFrame({"case_id": ["a", "b"], "x": [1., 2.], "y": [2., 3.]})
    path = tmp_path / "failed.png"
    with pytest.raises(OSError, match="disk failure"):
        if kind == "scatter":
            _scatter(data, "x", "y", "scatter", path)
        else:
            _case_bar(data, value_column="x", title="bar", ylabel="x", destination=path)
    assert set(plt.get_fignums()) == original
    assert not path.exists()


def test_satellite_preview_requires_documented_band_order(tmp_path):
    path = tmp_path / "patch.npy"
    np.save(path, np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4))
    rgb = _rgb_preview(path, ["blue", "red", "green"])
    assert rgb.shape == (4, 4, 3)
    assert np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1
    assert _rgb_preview(path, ["red", "green"]) is None
    assert _rgb_preview(path, ["red", "red", "blue"]) is None
    assert _rgb_preview(path, ["red", "green", "nir"]) is None
    broken = tmp_path / "broken.npy"
    broken.write_bytes(b"bad array")
    data = pd.DataFrame({
        "case_id": ["a", "broken", "missing", "ignored"],
        "image_path": [str(path), str(broken), str(tmp_path / "missing.npy"), str(path)],
        "sentinel_status": ["ready", "ready", "ready", "no_scene"],
    })
    destination = tmp_path / "preview.png"
    assert save_satellite_input_preview(data, destination, bands=["blue", "red", "green"]) == destination
    assert destination.read_bytes().startswith(b"\x89PNG")
    assert save_satellite_input_preview(data.iloc[1:3], tmp_path / "none.png",
                                        bands=["blue", "red", "green"]) is None
