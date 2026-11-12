import json
import os
from pathlib import Path
import subprocess
import sys


def test_warning_capture_deduplicates_repeated_warning(tmp_path: Path):
    env = os.environ.copy()
    env["LITHIUMSCOPE_LOG_DIR"] = str(tmp_path)
    code = """
import json
import warnings
from lithiumscope.core.logger import configure_logging, finalize_logging, install_warning_capture, warning_summary
configure_logging()
install_warning_capture()
warnings.simplefilter("always")
for _ in range(5):
    warnings.warn("repeated test warning", UserWarning)
print(json.dumps(warning_summary()))
finalize_logging()
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, env=env, timeout=20)
    summary = json.loads(result.stdout.strip())
    assert summary["unique"] == 1
    assert summary["total"] == 5
    assert summary["repeated"] == 4
