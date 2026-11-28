"""Use the explicitly approved EDA checkpoint, not an arbitrary newest run."""
from pathlib import Path
import json

from lithiumscope.core.hashing import file_sha256
from lithiumscope.results.eda_artifacts import verify_eda
from ..source_studies import historical_evidence, source_catalog

EDA_RUN = "eda_20261128T113124_016337Z_5451af22"
EDA_SIGNATURE = "81df1b4f5adc451b82024367a0b78feb3c200705bb76aaf23057d478287a86f8"
INVENTORY_RUN = "eda_20261128T113220_328539Z_f771bdcf"
INVENTORY_SHA = "cbed2c7bccc8239ea22ef643f1f066b0e928922fbb4a4597fd0dcc859eae561d"
BASE_SHA = "94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_file(path, digest):
    if Path(path).is_symlink() or file_sha256(path) != digest:
        raise ValueError(f"Evidence identity differs: {path}")
    return Path(path)


def load_checkpoint(source, cfg, root, research_root):
    """Relocate by registered run IDs and basenames; never follow stored absolute paths."""
    root, research_root = Path(root), Path(research_root)
    checked_file(source, BASE_SHA)
    eda = root / "model_1" / "eda" / EDA_RUN
    manifest = read(eda / "manifest.json")
    if (manifest.get("source_sha256") != BASE_SHA or manifest.get("model_group") != "model_1"
            or manifest.get("effective_config") != cfg or not verify_eda(eda, EDA_SIGNATURE)):
        raise ValueError("Approved canonical EDA is missing, changed or incompatible")
    inv_path = root / "model_1/source_catalog/executions" / INVENTORY_RUN / "source_inventory.json"
    inventory = read(checked_file(inv_path, INVENTORY_SHA))
    checked_file(inv_path.parent / "source_comparison.csv", inventory["source_comparison_sha256"])
    if (inventory["canonical_sha256"] != BASE_SHA or inventory["state"] != "complete"
            or inventory["model_fits"] != 0 or inventory["new_rows_approved"] != 0):
        raise ValueError("Approved source inventory incompatible")
    if tuple(r["source"] for r in inventory["active"]) != source_catalog.known_keys():
        raise ValueError("Approved source registry changed")
    profiles, issues = {}, {}
    for record in inventory["active"]:
        key = record["source"]
        try:
            run_id = str(record["eda_path"]).replace(chr(92), "/").rsplit("/", 1)[-1]
            if not run_id.startswith("eda_") or Path(run_id).name != run_id:
                raise ValueError("Invalid candidate run ID")
            path = root / "model_1/candidate_eda" / key / run_id
            mp = checked_file(path / "manifest.json", record["manifest_sha256"])
            profile = read(mp)
            if (profile.get("source_key") != key or profile.get("base_sha256") != BASE_SHA
                    or profile.get("all_originals_available") is not True
                    or not verify_eda(path, profile["signature"])):
                raise ValueError("Candidate profile incompatible")
            expected = {f["file"]: f["sha256"] for f in record["files"]}
            if profile["source_sha256"] != expected:
                raise ValueError("Candidate originals differ from approved inventory")
            folder = research_root / key if key.startswith("georoc_") else research_root / "directed" / key
            for name, digest in expected.items():
                if Path(name).name != name or "/" in name or chr(92) in name:
                    raise ValueError("Invalid original name")
                checked_file(folder / name, digest)
            history = historical_evidence.inspect_history(key, root, BASE_SHA)
            if not history or any(h["state"] != "verified_historical" or
                    h["canonical_source_matches"] is False for h in history):
                raise ValueError("Historical decision evidence incomplete")
            profiles[key] = {"path": path, "manifest": profile, "record": record,
                             "history": history, "original_folder": folder}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues[key] = f"{type(exc).__name__}: {exc}"
    return eda, manifest, profiles, issues
