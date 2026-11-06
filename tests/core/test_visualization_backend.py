import gc
import threading

from lithiumscope.core.visualization import (
    configure_headless_matplotlib,
    current_backend,
)
from lithiumscope.model_1.evaluation.plots import save_target_distribution


def test_matplotlib_is_forced_to_headless_agg(tmp_path):
    backend = configure_headless_matplotlib()

    assert backend.lower() == "agg"
    assert current_backend().lower() == "agg"

    destination = tmp_path / "plot.png"
    save_target_distribution([1, 2, 3, 4], destination)
    assert destination.exists()


def test_plot_cleanup_is_safe_from_worker_gc(tmp_path):
    configure_headless_matplotlib()
    destination = tmp_path / "plot.png"
    save_target_distribution([1, 2, 3, 4], destination)

    worker = threading.Thread(target=gc.collect)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert current_backend().lower() == "agg"
