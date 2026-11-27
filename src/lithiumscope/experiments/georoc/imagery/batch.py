"""Versioned exploratory imagery; uses registered acquisition and verified model inference."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
from time import monotonic
from uuid import uuid4

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger, run_log
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.experiments.scenarios.imagery import image_catalog
from lithiumscope.experiments.scenarios.verified_imagery import evaluate_verified_image
from lithiumscope.persistence.load_model import latest_model_path, load_latest_model
from lithiumscope.results.atomic_exports import write_json, write_table
from lithiumscope.tools.acquisition.sentinel2.configuration import sentinel_config
from lithiumscope.tools.acquisition.sentinel2.provider import Sentinel2Provider, SentinelProviderError
from lithiumscope.tools.acquisition.sentinel2.registered_patch import (
    download_registered_patch, validate_interval,
)
from .coordinate_plan import coordinate_plan
from ..protocol import EXPERIMENT_ROOT

OUT = RESULTS_DIR / "scientific_simulations" / "georoc_registered_v1"
LOGGER = get_logger("experiments.georoc.imagery")
COLUMNS = ["image_id", "latitude", "longitude", "state", "path", "scene_id",
           "baseline_status", "scenario_rows", "reason"]


def _revision():
    package = Path(__file__).resolve().parents[3]
    roots = ("experiments/georoc", "experiments/scenarios", "tools/acquisition/sentinel2",
             "datasets/adapters", "model_2/preparation", "model_2/prediction", "model_2/methodology/spectral")
    packages={}
    for name in ("numpy","pandas","scikit-learn","rasterio","pystac-client"):
        try:
            packages[name]=version(name)
        except PackageNotFoundError:
            packages[name]="missing"
    return canonical_json_hash({"packages":packages,"code":{p.relative_to(package).as_posix(): file_sha256(p)
        for folder in roots for p in sorted((package / folder).rglob("*.py"))}})


def _hashes(site: Path) -> dict:
    names = ("patch.npy", "patch.json", "patch.scl.npy", "patch.scene-audit.json", "image_scenario_metrics.csv")
    return {name: file_sha256(site / name) for name in names}


def _verify_scores(path: Path, seed: int):
    table = pd.read_csv(path)
    if table.scenario.tolist() != [s.name for s in image_catalog(seed)]:
        raise ValueError("Scenario output is incomplete or has changed ordering")
    if "status" not in table or "score" not in table:
        raise ValueError("Scenario output lacks status or score")
    scored = table.status.eq("SCORED")
    if not pd.to_numeric(table.loc[scored, "score"], errors="coerce").between(0, 1).all():
        raise ValueError("Invalid scenario score")
    return str(table.iloc[0].status), len(table)


def _recover(site: Path):
    if site.is_symlink():
        raise ValueError("Site directory may not be a symbolic link")
    preserved = site.parent.parent / ".quarantine"
    if preserved.is_symlink():
        raise ValueError("Quarantine may not be a symbolic link")
    preserved.mkdir(exist_ok=True)
    site.rename(preserved / (site.name + "." + uuid4().hex))


def _site(site, row, model, provider, identity, seed, *, recover):
    if site.is_symlink():
        raise ValueError("Site directory may not be a symbolic link")
    record_path = site / "result.json"
    patch = site / "patch.npy"
    signature = canonical_json_hash({"run": identity, "site": row.image_id})
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["signature"] != signature or record["hashes"] != _hashes(site):
                raise ValueError("Site output integrity mismatch")
            baseline, count = _verify_scores(site / "image_scenario_metrics.csv", seed)
            if (baseline, count) != (record["baseline_status"], record["scenario_rows"]):
                raise ValueError("Site score evidence mismatch")
            LOGGER.info("Reused verified site %s", row.image_id)
            return record
        except (OSError, ValueError, KeyError, TypeError):
            if not recover:
                raise
            _recover(site)
    elif site.exists() and any(site.iterdir()):
        # An interrupted scoring attempt can resume a valid registered patch.
        try:
            if patch.exists() or patch.with_suffix(".json").exists() or patch.with_suffix(".scl.npy").exists():
                download_registered_patch(row.image_id, row.latitude, row.longitude, patch, provider, signature)
        except (OSError, ValueError, RuntimeError):
            if not recover:
                raise
            _recover(site)
    site.mkdir(parents=True, exist_ok=True)
    path, meta = download_registered_patch(row.image_id, row.latitude, row.longitude, patch, provider, signature)
    scores = evaluate_verified_image(model, path, site, seed=seed)
    baseline, count = _verify_scores(scores, seed)
    record = {"signature": signature, "hashes": _hashes(site), "baseline_status": baseline,
              "scenario_rows": count, "scene_id": meta["scene_id"]}
    write_json(record_path, record)
    return record


def run(source: Path, *, root: Path = OUT, window="2025-01-01/2025-12-31",
        minimum_distance_km=100, decimals=4, seed=42, limit=0,
        acquire=False, recover=False, provider_factory=Sentinel2Provider):
    source, root = Path(source).resolve(), Path(root).resolve()
    validate_interval(window)
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("Invalid site limit")
    digest = file_sha256(source)
    sites, ledger = coordinate_plan(pd.read_csv(source), minimum_distance_km=minimum_distance_km, decimals=decimals)
    if file_sha256(source) != digest:
        raise ValueError("Candidate source changed during planning")
    if limit:
        sites = sites.head(limit)
        outside=ledger.plan_state.eq("selected") & ~ledger.image_id.isin(sites.image_id)
        ledger.loc[outside,"plan_state"]="not_selected"
        ledger.loc[outside,"plan_reason"]="site_limit"
    summary = {"state": "preflight", "sites": len(sites), "candidate_rows": len(ledger),
               "source_sha256": digest, "independent_accuracy_certified": False}
    if not acquire:
        return summary
    if sites.empty:
        return {**summary, "state": "no_eligible_sites"}
    model_path = latest_model_path("model_2")
    model_hash = file_sha256(model_path)
    model, actual = load_latest_model("model_2")
    if actual != model_path or file_sha256(actual) != model_hash:
        raise ValueError("Active model changed during loading")
    config = replace(sentinel_config(load_config("model_2")), datetime=window)
    policy = {"source_sha256": digest, "model_sha256": model_hash, "imagery": asdict(config),
              "minimum_distance_km": minimum_distance_km, "decimals": decimals, "seed": seed,
              "limit": limit, "revision": _revision(), "sites": sites.to_dict("records")}
    identity = canonical_json_hash(policy)
    directory = root / identity
    if directory.is_symlink():
        raise ValueError("Experiment directory may not be a symbolic link")
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("Experiment already running or interrupted; inspect its lock") from exc
    started = monotonic()
    rows = []
    def checkpoint(state):
        write_table(directory / "patch_manifest.csv", pd.DataFrame(rows, columns=COLUMNS))
        result = {**summary, "state": state, "version_id": identity, "directory": str(directory),
                  "processed": len(rows), "scored": sum(r.get("baseline_status") == "SCORED" for r in rows),
                  "failed": sum(r["state"] == "error" for r in rows), "elapsed_seconds": monotonic()-started}
        write_json(directory / "status.json", result)
        return result
    try:
        plan_file=directory / "plan.json"
        if plan_file.is_file() and canonical_json_hash(json.loads(plan_file.read_text(encoding="utf-8")))!=identity:
            raise ValueError("Experiment plan identity mismatch")
        write_json(plan_file, policy)
        write_table(directory / "coordinate_plan.csv", sites)
        write_table(directory / "candidate_ledger.csv", ledger)
        provider = provider_factory(config)
        checkpoint("running")
        for row in sites.itertuples(index=False):
            if (directory / "STOP").exists():
                return checkpoint("stopped")
            site = directory / "sites" / row.image_id
            current = {"image_id": row.image_id, "latitude": row.latitude, "longitude": row.longitude}
            try:
                record = _site(site, row, model, provider, identity, seed, recover=recover)
                current.update(state="verified", path=(site / "patch.npy").relative_to(directory).as_posix(),
                               scene_id=record["scene_id"], baseline_status=record["baseline_status"],
                               scenario_rows=record["scenario_rows"])
            except SentinelProviderError:
                raise
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                current.update(state="error", reason=f"{type(exc).__name__}: {exc}")
                LOGGER.warning("Site failed %s: %s", row.image_id, exc)
            rows.append(current)
            checkpoint("running")
        if file_sha256(source) != digest or file_sha256(model_path) != model_hash:
            raise ValueError("Protected experiment inputs changed")
        from .analysis import summarize_registered
        summarize_registered(directory)
        return checkpoint("complete" if all(r["state"] == "verified" for r in rows) else "incomplete")
    except BaseException:
        checkpoint("interrupted")
        raise
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=EXPERIMENT_ROOT / "full_mamani_contamination_audit.csv")
    parser.add_argument("--window", default="2025-01-01/2025-12-31")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    with run_log("simulations", "georoc_registered_imagery"):
        print(run(args.source, window=args.window, limit=args.limit, acquire=args.acquire, recover=args.recover))


if __name__ == "__main__":
    main()
