"""Synthetic CPU/accelerator benchmark. Explicit --execute; no external network."""
import argparse
from pathlib import Path
from lithiumscope.resources.monitoring.acceleration_benchmark import compare

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    print(compare(args.output,execute=args.execute))
