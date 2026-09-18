from pathlib import Path

from lithiumscope.core.device import DeviceInfo
from lithiumscope.model_2.training.competition import run_model_2_competition


def train_model_2(manifest_path: Path, device: DeviceInfo):
    """Public training entry point: run the complete Model 2 competition."""
    return run_model_2_competition(manifest_path, device)
