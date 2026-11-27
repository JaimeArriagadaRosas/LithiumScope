"""Explicit local-only, immutable GEOROC screening and evidence pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger, run_log
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.datasets.management.snapshots import dataset_implementation_revision, materialize_snapshot, verify_snapshot
from lithiumscope.experiments.georoc.screening import andean_candidates, provenance_screen, extended_evidence, mamani_overlap
from lithiumscope.experiments.reporting import georoc_evidence
from lithiumscope.results.atomic_exports import write_json

OUT = RESULTS_DIR / "scientific_simulations" / "georoc_evidence_v1"


def run(partitions, training, mamani, citations, *, root=OUT, execute=False, recover=False):
    root = Path(root)
    partitions = [Path(p).resolve() for p in partitions]
    if not partitions or len({p.name for p in partitions}) != len(partitions):
        raise ValueError("Declare uniquely named source partitions")
    sources = {f"partition:{p.name}": p for p in partitions}
    sources.update(training=Path(training), mamani=Path(mamani), citations=Path(citations))
    missing = [key for key, path in sources.items() if not path.is_file()]
    if missing:
        return {"state":"blocked","missing":missing,"external_accuracy_certified":False}
    hashes = {key:file_sha256(path) for key,path in sources.items()}
    if not execute:
        return {"state":"preflight","source_hashes":hashes,"external_accuracy_certified":False}
    logger = get_logger("experiments.georoc.workflow")
    revision = dataset_implementation_revision("georoc-evidence-v1", dataset_root=Path(__file__).parent)
    revision += dataset_implementation_revision("dataset-inputs")
    revision += ":" + file_sha256(Path(georoc_evidence.__file__))
    versions = {}
    def stage(name, inputs, call, filename):
        def transform(paths, output):
            call(paths, output.parent)
            (output.parent/filename).replace(output)
        result = materialize_snapshot(root/name, inputs, policy={"stage":name,"external_accuracy_certified":False},
                                      transform_revision=revision, transform=transform, recover_corrupt=recover)
        verify_snapshot(result.directory)
        versions[name]={"version_id":result.version_id,"directory":str(result.directory),"reused":result.reused}
        logger.info("GEOROC stage %s version=%s reused=%s",name,result.version_id,result.reused)
        return result.data_path
    raw = {k:v for k,v in sources.items() if k.startswith("partition:")}
    candidate = stage("candidates",{**raw,"training":sources["training"]},
        lambda paths, out: andean_candidates.run(out,paths["training"],partitions=[v for k,v in paths.items() if k.startswith("partition:")]),
        "georoc_andean_candidates.csv")
    screen = stage("screening",{"candidate":candidate},
        lambda paths,out: provenance_screen.main(out,candidate_path=paths["candidate"]),
        "georoc_provenance_screen.csv")
    evidence = stage("provenance",{**raw,"candidate":screen,"citations":sources["citations"]},
        lambda paths,out: extended_evidence.run(out,candidate_path=paths["candidate"],citations=paths["citations"],
            partitions=[v for k,v in paths.items() if k.startswith("partition:")]),
        "georoc_extended_evidence_screen.csv")
    overlap = stage("overlap",{"candidate":evidence,"mamani":sources["mamani"],"training":sources["training"]},
        lambda paths,out: mamani_overlap.main(out,candidate_path=paths["candidate"],mamani_path=paths["mamani"],training_path=paths["training"]),
        "full_mamani_contamination_audit.csv")
    report = stage("report",{"candidate":overlap,"citations":sources["citations"]},
        lambda paths,out: georoc_evidence.main(out,candidate_path=paths["candidate"],citations_path=paths["citations"]),
        "external_validation_decision_ledger.csv")
    if hashes!={key:file_sha256(path) for key,path in sources.items()}:
        raise ValueError("Original sources changed during the workflow")
    identity=canonical_json_hash({"sources":hashes,"revision":revision})
    summary={"state":"complete","version_id":identity,"source_hashes":hashes,"stages":versions,
             "imagery_input":str(overlap),"report":str(report),"external_accuracy_certified":False}
    write_json(root/"runs"/(identity+".json"),summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition",type=Path,action="append",required=True)
    parser.add_argument("--training",type=Path,required=True)
    parser.add_argument("--mamani",type=Path,required=True)
    parser.add_argument("--citations",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--recover",action="store_true")
    args=parser.parse_args()
    with run_log("simulations","georoc_evidence_pipeline"):
        print(run(args.partition,args.training,args.mamani,args.citations,execute=args.execute,recover=args.recover))


if __name__=="__main__":
    main()
