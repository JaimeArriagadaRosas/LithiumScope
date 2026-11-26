"""Standalone saved-model comparison: no simulations, fit, download or activation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from uuid import uuid4

from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.experiments.shared.comparison_models import discover_models
from lithiumscope.experiments.shared.saved_comparison import compare_saved
from lithiumscope.results.atomic_exports import write_legacy_evidence_json


def run_version_comparison(*,output_root=None):
    root=output_root or RESULTS_DIR/"scientific_simulations"/"version_comparisons"
    destination=root/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    destination.mkdir(parents=True,exist_ok=False)
    records={}
    from lithiumscope.results.reference_benchmarks import OASIS_M1
    for group in ("model_1","model_2"):
        available,issues=discover_models(group)
        record={"models":{role:identity for role,(_,identity) in available.items()},
                "inventory_issues":issues,"state":"blocked"}
        # Always distinguish descriptive stored metrics from strict paired evidence.
        record["descriptive_metrics"]={}
        for role,(_,identity) in available.items():
            from pathlib import Path
            try:
                metadata=json.loads(Path(identity["path"]).with_name("metadata.json").read_text(encoding="utf-8"))
                record["descriptive_metrics"][role]={"samples":metadata.get("samples"),"metrics":metadata.get("metrics",{}),
                    "dataset_sha256":identity.get("dataset_sha256"),"validation_strategy":metadata.get("validation_strategy"),
                    "scope":"own_training_protocol"}
            except (OSError,ValueError,KeyError):
                continue
        try:
            path=compare_saved(group,available,destination/group)
            record.update(state="complete",comparison_path=str(path),
                          comparison=json.loads(path.read_text(encoding="utf-8")))
        except (FileNotFoundError,ValueError,KeyError,OSError) as exc:
            record["reason"]=str(exc)
        if group=="model_2" and "candidate" in available:
            from lithiumscope.experiments.shared.m2_recovery import recover_m2_reports
            try:
                record["report_recovery"]=recover_m2_reports(available["candidate"][1],destination/group/"recovered_reports")
            except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
                record["report_recovery"]={"state":"partial","reason":str(exc)}
            print("M2 informes históricos:",record["report_recovery"]["state"],flush=True)
        records[group]=record
        print(f"{group}: {record['state']} {record.get('reason','')}",flush=True)
    manifest={"state":"complete" if all(r["state"]=="complete" and r.get("report_recovery",{}).get("state","complete")=="complete" for r in records.values()) else "partial",
              "scope":"saved_oof_only_development","independent_holdout":False,"models":records,
              "academic_reference_m1":OASIS_M1,
              "training":False,"promotion":False,"external_acquisition":False}
    return write_legacy_evidence_json(destination/"manifest.json",manifest)
