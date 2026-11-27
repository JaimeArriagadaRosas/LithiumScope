"""Explicit research-only candidate workflow; never changes configured training inputs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from contextlib import nullcontext
import json
from pathlib import Path

import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.adapters.georoc.ingestion import ingest_georoc_files
from lithiumscope.datasets.adapters.registry import get_adapter
from lithiumscope.datasets.adapters.spectral.archives import selected_text_member, list_text_members
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
        with (output.parent / "raw_exclusions.csv").open("w", encoding="utf-8", newline="") as handle, \
             (output.parent / "analytical_censoring.jsonl").open("w", encoding="utf-8") as censoring:
            writer = csv.DictWriter(handle, fieldnames=["source_file", "source_row_position", "reason"])
            writer.writeheader()
            def record_censoring(records):
                for record in records:
                    censoring.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            ingest_georoc_files([path], output, audit_path=output.parent / "ingestion_audit.json",
                                exclusion_callback=writer.writerows, censoring_callback=record_censoring, **ingestion_policy)
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


def adapt_reference(source: Path, *, key: str, options: dict, root: Path,
                    member: str | None = None, recover_corrupt: bool = False):
    """Version a local reference file, without changing the training data."""
    source = Path(source)
    if key not in {"mamani09_public_mirror", "fregeneda_almendra", "greenpeg", "sentinel2"}:
        raise ValueError("Use the existing georoc workflow for multi-source GEOROC")
    if member is not None and key not in {"fregeneda_almendra", "greenpeg"}:
        raise ValueError("Archive members are supported only for reference spectra")
    inputs = {"source": source}
    if key == "sentinel2":
        sidecars = [p for p in (source.with_suffix(".json"), source.with_suffix(source.suffix+".json")) if p.is_file()]
        if len(sidecars) != 1:
            raise ValueError("Exactly one patch sidecar required")
        inputs["sidecar"] = sidecars[0]
        if json.loads(sidecars[0].read_text(encoding="utf-8")).get("scl_sha256") is not None:
            inputs["scl"] = source.with_suffix(".scl.npy")
    adapter = get_adapter(key)
    def transform(inputs, output):
        archive = inputs["source"]
        context = selected_text_member(archive, member) if member is not None else nullcontext(archive)
        with context as path:
            if key == "sentinel2":
                _, evidence = adapter(path, **options)
                frame = pd.DataFrame({"band": evidence["bands"], "valid_fraction": evidence["band_valid_fractions"]})
            else:
                result = adapter(path, **options)
                frame, evidence = result.frame, dict(result.evidence)
        evidence.update(source_key=key, archive_member=member, input_sha256=file_sha256(archive),
                        training_label_certified=False)
        if key == "mamani09_public_mirror":
            _export_quality(frame, candidate_policy("model_1"), output)
        else:
            frame.to_csv(output, index=False)
        (output.parent / "adapter_evidence.json").write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return materialize_snapshot(root, inputs, policy={"source": key, "options": options, "member": member,
                                    "source_name": source.name}, transform_revision=dataset_implementation_revision("source-adapter-v1"),
                                transform=transform, recover_corrupt=recover_corrupt)


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
    adapt = commands.add_parser("adapt")
    adapt.add_argument("key", choices=("mamani09_public_mirror","fregeneda_almendra","greenpeg","sentinel2"))
    adapt.add_argument("source", type=Path)
    adapt.add_argument("--options", type=Path, help="JSON file containing explicit adapter parameters")
    adapt.add_argument("--member", help="Exact member of the downloaded spectral ZIP")
    adapt.add_argument("--recover-corrupt", action="store_true")
    members = commands.add_parser("members")
    members.add_argument("archive", type=Path)
    commands.add_parser("list")
    verify = commands.add_parser("verify")
    verify.add_argument("directory", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--identity", nargs="+", required=True)
    args = parser.parse_args()
    root = RESULTS_DIR / "dataset_candidates"
    if args.command == "members":
        print(json.dumps(list_text_members(args.archive), indent=2, ensure_ascii=False))
        return
    if args.command == "list":
        print(json.dumps(list_snapshots(root), indent=2, ensure_ascii=False))
        return
    if args.command == "verify":
        directory = (root / args.directory).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError("Candidate version outside catalog")
        print(json.dumps(verify_snapshot(directory), indent=2, ensure_ascii=False))
        return
    if args.command == "adapt":
        options = json.loads(args.options.read_text(encoding="utf-8")) if args.options else {}
        if not isinstance(options, dict):
            raise ValueError("Adapter options must be a JSON object")
        result = adapt_reference(args.source, key=args.key, options=options, root=root / "adapted" / args.key,
                                 member=args.member, recover_corrupt=args.recover_corrupt)
    elif args.command == "audit":
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
