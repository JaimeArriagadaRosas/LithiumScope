"""Historical published-partition experiment: acquisition requires --acquire."""
from pathlib import Path
import argparse
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.tools.acquisition.georoc.dataverse import fetch_partition
from lithiumscope.results.atomic_exports import write_json

PARTITIONS=(219128,219105,219094)


def run(output,*,acquire=False):
    output=Path(output)
    plan=[{"identifier":ident,"destination":f"2026-09-PVFZCE_ANDEAN_ARC_part{i+1}.csv"} for i,ident in enumerate(PARTITIONS)]
    if not acquire:
        return {"state":"preflight","partitions":plan}
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for item in plan:
        records.append(fetch_partition(item["identifier"],output/item["destination"]))
        write_json(output/"download_manifest.json",{"scope":"published historical experiment","partitions":records})
    return {"state":"complete","partitions":records}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=RESULTS_DIR/"scientific_simulations/external_geochemistry_audit/georoc_andean_raw")
    parser.add_argument("--acquire",action="store_true")
    args=parser.parse_args()
    print(run(args.output,acquire=args.acquire))


if __name__=="__main__":
    main()
