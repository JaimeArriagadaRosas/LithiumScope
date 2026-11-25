"""Opt-in paired synthetic local pipeline benchmark."""
import argparse
from pathlib import Path
from lithiumscope.resources.monitoring.pipeline_benchmark import run

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--rows",type=int,default=240)
    args=parser.parse_args()
    print(run(args.output,execute=args.execute,rows=args.rows))
