"""Opt-in grouped uncertainty experiment; default hyperparameters, no model promotion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.shared.design import choose_algorithms
from lithiumscope.model_1.evaluation.group_intervals import run_group_interval_study
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.model_1.training.preprocessing.spatial_tabnet import fit_regression_estimator
from lithiumscope.resources.detection.device import detect_device


def run(output: Path, *, algorithms=None, execute=False):
    config = load_config("model_1")
    selected = choose_algorithms(config["model"]["algorithms"], algorithms)
    seed = int(config["validation"]["random_seed"])
    source = local_training_source("model_1")
    data = prepare_training_data(source)
    output.mkdir(parents=True, exist_ok=False)
    device = detect_device(prefer_gpu=False)
    reports = []
    for algorithm in selected:
        def fit(x,y,groups):
            estimator = build_pipeline(algorithm,device,seed,data.schema,{},target_transform="identity")
            return fit_regression_estimator(estimator,x,y,groups)
        try:
            report, table = run_group_interval_study(data.x,data.y,data.groups,fit=fit,
                predict=lambda m,x:m.predict(x), seed=seed,execute=execute)
            if table is not None:
                table.to_csv(output/f"{algorithm}_intervals.csv",index=False)
        except Exception as exc:
            report = {"state":"failed","error_type":type(exc).__name__}
        reports.append({"algorithm":algorithm,**report})
        (output/"manifest.json").write_text(json.dumps({"source_sha256":file_sha256(source),
            "reports":reports,"hyperparameters":"defaults","automatic_selection":False},
            indent=2,allow_nan=False),encoding="utf-8")
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm",action="append")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    output=RESULTS_DIR/"model_1"/"methodology"/("group_uncertainty_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8])
    run(output,algorithms=args.algorithm,execute=args.execute)
    print(output)


if __name__=="__main__":
    main()
