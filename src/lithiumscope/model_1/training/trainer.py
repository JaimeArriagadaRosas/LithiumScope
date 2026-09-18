from pathlib import Path

from lithiumscope.core.device import DeviceInfo
from lithiumscope.model_1.training.competition import run_model_1_competition


def train_model_1(path: Path, device: DeviceInfo):
    """Public training entry point: run the complete Model 1 competition."""
    return run_model_1_competition(path, device)
