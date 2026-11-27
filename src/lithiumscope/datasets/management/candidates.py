"""Explicit research-only candidate workflow; never changes configured training inputs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.adapters.georoc.ingestion import ingest_georoc_files
from lithiumscope.datasets.management.snapshots import (dataset_implementation_revision, list_snapshots,
    materialize_snapshot, materialize_sources, verify_snapshot)
from lithiumscope.datasets.provenance.comparison import compare_versions
from lithiumscope.datasets.provenance.identity import observation_ids
from lithiumscope.datasets.readers.tabular import load_data
from lithiumscope.datasets.validation.contracts import QualityPolicy, candidate_policy
from lithiumscope.datasets.validation.quality_report import assess_quality, write_quality_report

# Bump whenever output-affecting workflow logic changes.
REVISION = "candidate-workflow-v2"


def _export_quality(frame: pd.DataFrame, policy: QualityPolicy, output: Path) -> None:
    report = assess_quality(frame, policy)
    frame.to_csv(output, index=False)
    write_quality_report(report, output.parent)
    for name, subset in report.partition(frame).items():
        subset.to_csv(output.parent / f"{name}.csv", index=False)


def audit_candidate(source: Path, *, root: Path, policy: QualityPolicy, recover_corrupt: bool = False):
    source = Path(source)
    def transform(inputs, output):
        path = inputs["source"]
        frame = load_data(path, quiet=True)
        if "audit_observation_id" in frame:
            raise ValueError("Reserved audit observation field already exists")
        positions = pd.DataFrame({"source_row_position": range(len(frame))}, index=frame.index)
        frame["audit_observation_id"] = observation_ids(
            positions, source=source.name, source_sha256=file_sha256(path))
        _export_quality(frame, policy, output)
    return materialize_snapshot(root, {"source": source}, policy={**asdict(policy), "source_name": source.name},
                                transform_revision=dataset_implementation_revision(REVISION), transform=transform,
                                recover_corrupt=recover_corrupt)


def build_georoc_candidates(sources: list[Path], *, root: Path, policy: QualityPolicy,
                            minimum_predictors: int = 8, iron_total_mode: str = "original",
                            recover_corrupt: bool = False):
    if policy.model_group != "model_1":
        raise ValueError("GEOROC candidate workflow is restricted to M1")
    named = {Path(path).name: Path(path) for path in sources}
    if not named or len(named) != len(sources):
        raise ValueError("Declare unique GEOROC source filenames")
    ingestion_policy = {"minimum_predictors": minimum_predictors, "iron_total_mode": iron_total_mode}
    def transform(inputs, output):
        _, path = next(iter(inputs.items()))
        with (output.parent / "raw_exclusions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_file", "source_row_position", "reason"])
            writer.writeheader()
            ingest_georoc_files([path], output, audit_path=output.parent / "ingestion_audit.json",
                                exclusion_callback=writer.writerows, **ingestion_policy)
    revision = dataset_implementation_revision(REVISION)
    processed = materialize_sources(root / "sources", named, policy=ingestion_policy,
                                    transform_revision=revision + "-georoc", transform=transform, recover_corrupt=recover_corrupt)
    source_digests = {name: verify_snapshot(snapshot.directory)["identity"]["sources"][name]
                      for name, snapshot in processed.items()}
    def assemble(inputs, output):
        frames = []
        for name, path in inputs.items():
            frame = load_data(path, quiet=True)
            frame["audit_observation_id"] = observation_ids(
                frame, source=name, source_sha256=source_digests[name])
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True, sort=False)
        # Preserve observations for review. No automatic cross-source deduplication.
        _export_quality(combined, policy, output)
    return materialize_snapshot(root / "combined", {name: s.data_path for name, s in processed.items()},
                                policy={"quality": asdict(policy), "sources": {name: s.version_id for name, s in processed.items()}},
                                transform_revision=revision + "-combined", transform=assemble, recover_corrupt=recover_corrupt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("source", type=Path)
    audit.add_argument("--model", choices=("model_1", "model_2"), required=True)
    audit.add_argument("--recover-corrupt", action="store_true")
    georoc = commands.add_parser("georoc")
    georoc.add_argument("sources", type=Path, nargs="+")
    georoc.add_argument("--minimum-predictors", type=int, default=8)
    georoc.add_argument("--iron-total-mode", choices=("original", "feot_equivalent"), default="original")
    georoc.add_argument("--recover-corrupt", action="store_true")
    commands.add_parser("list")
    verify = commands.add_parser("verify")
    verify.add_argument("directory", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--identity", nargs="+", required=True)
    args = parser.parse_args()
    root = RESULTS_DIR / "dataset_candidates"
    if args.command == "list":
        print(json.dumps(list_snapshots(root), indent=2, ensure_ascii=False))
        return
    if args.command == "verify":
        directory = (root / args.directory).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError("Candidate version outside catalog")
        print(json.dumps(verify_snapshot(directory), indent=2, ensure_ascii=False))
        return
    if args.command == "audit":
        result = audit_candidate(args.source, root=root / "audit", policy=candidate_policy(args.model), recover_corrupt=args.recover_corrupt)
    elif args.command == "georoc":
        result = build_georoc_candidates(args.sources, root=root / "georoc",
                    policy=candidate_policy("model_1"), minimum_predictors=args.minimum_predictors,
                    iron_total_mode=args.iron_total_mode, recover_corrupt=args.recover_corrupt)
    else:
        before, after = (load_data(path, quiet=True) for path in (args.before, args.after))
        print(json.dumps(compare_versions(before, after, identity=tuple(args.identity)), indent=2, ensure_ascii=False))
        return
    print(json.dumps({"directory": str(result.directory), "version_id": result.version_id,
                      "reused": result.reused, "production_changed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
