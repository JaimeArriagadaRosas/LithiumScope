"""Local prospective holdout CLI. Does not activate a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lithiumscope.evaluation.holdout.registry import verified_registration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="operation",required=True)
    reg=sub.add_parser("register")
    reg.add_argument("root",type=Path)
    reg.add_argument("--model-group",choices=["model_1","model_2"],required=True)
    reg.add_argument("--sites-path",type=Path,required=True)
    reg.add_argument("--exposure-paths",type=Path,nargs="+",required=True)
    for arg in ("reference","candidate","review-path"):
        reg.add_argument("--"+arg,type=Path,required=True)
    reg.add_argument("--label-commitment-sha256",required=True)
    reg.add_argument("--minimum-distance-m",type=float,required=True)
    reg.add_argument("--label-cutoff-ppm",type=float)
    score=sub.add_parser("score")
    score.add_argument("root",type=Path)
    score.add_argument("--trusted-local-bundles",action="store_true",required=True)
    evaluate=sub.add_parser("evaluate")
    evaluate.add_argument("root",type=Path)
    evaluate.add_argument("--labels-path",type=Path,required=True)
    args=vars(parser.parse_args())
    operation=args.pop("operation")
    root=args.pop("root")
    group=args.pop("model_group") if operation=="register" else verified_registration(root)[0]["model_group"]
    if group=="model_1":
        from lithiumscope.model_1.pipeline import run_independent_evaluation
    elif group=="model_2":
        from lithiumscope.model_2.pipeline import run_independent_evaluation
    else:
        raise ValueError("Unknown registered model")
    result=run_independent_evaluation(operation,root,**args)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
