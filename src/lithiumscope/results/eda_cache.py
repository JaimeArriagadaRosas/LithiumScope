"""Cache the exact descriptive EDA inputs without rerunning preparation on a hit."""
from __future__ import annotations

import json
from pathlib import Path

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.core.run_resume import build_training_signature
from lithiumscope.core.logging.session import get_logger
from lithiumscope.results.atomic_exports import write_json
from lithiumscope.results.eda_artifacts import verify_eda


def key_for(group,source):
    from importlib.metadata import version
    if group not in {"model_1","model_2"}:
        raise ValueError("Unknown model group")
    cfg=load_config(group)
    if group=="model_1":
        upstream=build_training_signature(group,source,("app",group))
    else:
        from lithiumscope.model_2.preparation.features.v4_inputs import build_scientific_training_signature
        upstream=build_scientific_training_signature(source,cfg)
    package=Path(__file__).resolve().parents[1]
    relevant=(package/group,package/"core"/"scientific",package/"datasets",package/"results",package/"core"/"configuration")
    return canonical_json_hash({"upstream":upstream,"config":cfg,"source_sha256":file_sha256(source),
        "code":{p.relative_to(package).as_posix():file_sha256(p) for r in relevant for p in sorted(r.rglob("*.py"))},
        "versions":{name:version(name) for name in ("matplotlib","numpy","pandas","scikit-learn")}})


def index_path(group,key):
    return RESULTS_DIR/group/"eda_index"/(key+".json")


def lookup(group,source):
    key=key_for(group,source)
    path=index_path(group,key)
    if not path.is_file():
        return None
    try:
        record=json.loads(path.read_text(encoding="utf-8"))
        name=record["eda_run_id"]
        if not isinstance(name,str) or not name.startswith("eda_") or Path(name).name!=name or "/" in name or chr(92) in name:
            raise ValueError("Invalid EDA run identity")
        evidence=RESULTS_DIR/group/"eda"/name
        manifest=json.loads((evidence/"manifest.json").read_text(encoding="utf-8"))
        if (record["key"]==key and manifest.get("model_group")==group
                and manifest.get("source_sha256")==file_sha256(source)
                and manifest.get("training_reusable") is False
                and file_sha256(evidence/"manifest.json")==record["manifest_sha256"]
                and verify_eda(evidence,record["signature"])):
            return evidence
    except (OSError,ValueError,TypeError,KeyError):
        get_logger("results.eda_cache").warning("Invalid EDA index model=%s key=%s",group,key)
    return None


def record(group,source,evidence,key):
    evidence=Path(evidence)
    if key_for(group,source)!=key:
        raise ValueError("EDA inputs changed during analysis")
    manifest=json.loads((evidence/"manifest.json").read_text(encoding="utf-8"))
    if evidence.parent.resolve()!=(RESULTS_DIR/group/"eda").resolve() or manifest.get("model_group")!=group or manifest.get("source_sha256")!=file_sha256(source):
        raise ValueError("EDA source or owner mismatch")
    if manifest.get("training_reusable") is not False or not verify_eda(evidence,manifest["signature"]):
        raise ValueError("EDA integrity failure")
    write_json(index_path(group,key),{"key":key,"eda_run_id":evidence.name,
        "signature":manifest["signature"],"manifest_sha256":file_sha256(evidence/"manifest.json")})
    get_logger("results.eda_cache").info("Indexed EDA model=%s run=%s",group,evidence.name)
