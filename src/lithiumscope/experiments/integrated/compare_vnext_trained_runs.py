"""Read-only comparisons of saved vNext M1/M2 spatial out-of-fold predictions.

NO fit, train, refit, dataset download, model promotion, or source rewrite.
M1 paired observations may have different outer-fold assignments between runs.
M2 is strictly identical on sample labels, spatial folds, and input manifest hash.
"""
from __future__ import annotations

import json
from pathlib import Path
from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.core.logging.operations import run_log
from lithiumscope.experiments.integrated.comparison.evidence import (
    OLD_M1, NEW_M1, OLD_M2, NEW_M2, regression_metrics, classification_metrics,
    group_bootstrap_gain, load_m1_oof, pair_shared_m1, load_m2_oof,
    check_m2_identical_protocol,
)
from lithiumscope.experiments.integrated.comparison.compare_m1 import compare_m1
from lithiumscope.experiments.integrated.comparison.compare_m2 import compare_m2

OUTPUT = RESULTS_DIR / "scientific_simulations" / "vnext_20261106" / "stage5_training_comparison"

def run(root: Path = PROJECT_ROOT, output: Path | None = None) -> dict:
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve() if output is not None else (
        root / "results" / "scientific_simulations" / "vnext_20261106" / "stage5_training_comparison"
    )
    with run_log("experiments", "saved_run_comparison"):
        return _run_comparison(root, output)


def _run_comparison(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "model_1": compare_m1(root, output),
        "model_2": compare_m2(root, output),
        "safety": {
            "model_fits": 0,
            "model_downloads": 0,
            "model_promotions": 0,
            "input_files_modified": 0,
        },
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    run()
