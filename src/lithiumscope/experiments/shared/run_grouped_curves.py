"""Explicit local runner for both six-family learning-curve diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from uuid import uuid4

from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.experiments.shared.curve_suite import run_curve_suite


from lithiumscope.model_1.diagnostics.learning_curves import regression_inputs
from lithiumscope.model_2.diagnostics.learning_curves import classification_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("model_1", "model_2"))
    parser.add_argument("--execute", action="store_true", help="Fit diagnostic curves; never save a model.")
    parser.add_argument("--prefer-gpu", action="store_true")
    args = parser.parse_args()
    run = RESULTS_DIR / "diagnostics" / "grouped_learning_curves" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12])
    if args.model == "model_1":
        inputs = regression_inputs()
    else:
        inputs = classification_inputs(run / "preparation")
    device = detect_device(prefer_gpu=args.prefer_gpu)
    fit = inputs.pop("fit_builder")(device)
    report = run_curve_suite(**inputs, device=device, fit=fit, output=run / "curves", execute=args.execute)
    print(f"Curvas {args.model}: {report['state']} — {run}")


if __name__ == "__main__":
    main()
