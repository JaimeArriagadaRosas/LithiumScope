"""Copy prediction inputs and verify that the evidence did not change."""
from pathlib import Path
import shutil
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger

logger = get_logger("prediction.inputs")


def snapshot_input(source: Path, destination: Path) -> str:
    source, destination = Path(source), Path(destination)
    expected = file_sha256(source)
    shutil.copy2(source, destination)
    if file_sha256(destination) != expected or file_sha256(source) != expected:
        destination.unlink(missing_ok=True)
        raise ValueError(f"Prediction input changed during snapshot: {source}")
    logger.info("Verified input snapshot %s -> %s sha256=%s", source, destination, expected)
    return expected
