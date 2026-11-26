"""Cache the exact descriptive EDA inputs without rerunning preparation on a hit."""
from __future__ import annotations

import json
from pathlib import Path

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.experiments.generalization.prepared import identity
from lithiumscope.results.atomic_exports import write_json
from lithiumscope.results.eda_artifacts import verify_eda


def key_for(group,source):
    from importlib.metadata import version
    package=Path(__file__).resolve().parents[2]
    relevant=(package/group/"eda",package/"core"/"scientific",package/"results")
    return canonical_json_hash({"prepared":identity(group,source,load_config(group)),
        "code":{p.relative_to(package).as_posix():file_sha256(p) for r in relevant for p in sorted(r.rglob("*.py"))},
        "matplotlib":version("matplotlib")})


def index_path(group,key):
    return RESULTS_DIR/group/"eda_index"/(key+".json")


def lookup(group,source):
    key=key_for(group,source)
    path=index_path(group,key)
    if not path.is_file():
        return None
    try:
        record=json.loads(path.read_text(encoding="utf-8"))
        evidence=RESULTS_DIR/group/"eda"/record["eda_run_id"]
        if (record["key"]==key and Path(record["eda_run_id"]).name==record["eda_run_id"]
                and file_sha256(evidence/"manifest.json")==record["manifest_sha256"]
                and verify_eda(evidence,record["signature"])):
            return evidence
    except (OSError,ValueError,TypeError,KeyError):
        pass
    return None


def record(group,source,evidence,key):
    evidence=Path(evidence)
    if key_for(group,source)!=key:
        raise ValueError("EDA inputs changed during analysis")
    manifest=json.loads((evidence/"manifest.json").read_text(encoding="utf-8"))
    if not verify_eda(evidence,manifest["signature"]):
        raise ValueError("EDA integrity failure")
    write_json(index_path(group,key),{"key":key,"eda_run_id":evidence.name,
        "signature":manifest["signature"],"manifest_sha256":file_sha256(evidence/"manifest.json")})
