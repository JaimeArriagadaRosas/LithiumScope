"""Freeze historical outer and, where available, archived inner group folds."""
from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import GroupKFold

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import DATA_DIR


def validate_partition(a, b, universe, groups):
    a, b, universe = map(lambda v: np.asarray(v,dtype=int),(a,b,universe))
    if (not len(a) or not len(b) or len(np.unique(a))!=len(a) or len(np.unique(b))!=len(b)
            or np.intersect1d(a,b).size
            or not np.array_equal(np.sort(np.r_[a,b]),np.sort(universe))
            or set(groups.iloc[a]) & set(groups.iloc[b])):
        raise ValueError("Invalid or overlapping historical spatial partition")
    return a,b


def outer_from_manifest(fold_ids, groups, count):
    ids = np.asarray(fold_ids,dtype=int)
    if len(ids)!=len(groups) or set(ids)!=set(range(1,count+1)):
        raise ValueError("Incomplete historical outer folds")
    universe = np.arange(len(ids))
    return [validate_partition(universe[ids!=n],universe[ids==n],universe,groups)
            for n in range(1,count+1)]


def validate_plan(record, outer, groups, inner_count):
    if len(record)!=len(outer):
        raise ValueError("Different number of outer folds")
    result = []
    for n, ((tr,te), saved) in enumerate(zip(outer,record,strict=True),1):
        if (saved["fold"]!=n or not np.array_equal(saved["train"],tr)
                or not np.array_equal(saved["test"],te) or len(saved["inner"])!=inner_count):
            raise ValueError("Archived plan does not match frozen outer folds")
        inner = [validate_partition(row["train"],row["validation"],tr,groups) for row in saved["inner"]]
        if not np.array_equal(np.sort(np.concatenate([b for _,b in inner])),np.sort(tr)):
            raise ValueError("Archived inner validation coverage incomplete")
        result.append((tr,te,inner))
    return result


def find_plan(snapshot, source_hash, outer, groups, inner_count):
    """Only accept hash-verified matching prepared bytes and complete partitions."""
    prepared_hash = file_sha256(snapshot/"prepared.json")
    root = DATA_DIR/"processed"/"model_1"/"prepared_versions"
    matches = {}
    for path in root.glob("*/treatment_plans/*/manifest.json"):
        try:
            plan = json.loads(path.read_text(encoding="utf-8"))
            old_snapshot = path.parents[2]
            old = json.loads((old_snapshot/"manifest.json").read_text(encoding="utf-8"))
            partition = path.with_name("partitions.json")
            digest = file_sha256(partition)
            if (old["source_sha256"]!=source_hash or old["artifacts"]["prepared.json"]!=prepared_hash
                    or file_sha256(old_snapshot/"prepared.json")!=prepared_hash
                    or plan["prepared_identity"]!=old_snapshot.name
                    or plan["identity"]!=path.parent.name
                    or plan["state"] not in {"complete","partial"}
                    or plan["artifacts"]["partitions.json"]!=digest):
                continue
            splits = validate_plan(json.loads(partition.read_text(encoding="utf-8")),outer,groups,inner_count)
            matches[digest]=(splits,str(partition))
        except (OSError,ValueError,KeyError,TypeError,IndexError):
            continue
    if len(matches)>1:
        raise ValueError("Multiple incompatible archived inner plans; select a unique historical plan")
    if matches:
        digest,(splits,path)=next(iter(matches.items()))
        return splits,{"inner_protocol":"verified_archived_plan","partition_sha256":digest,"partition_path":path}
    # Outer assignment remains exact. Inner CV is still spatial and restricted to outer TRAIN.
    splits=[]
    for tr,te in outer:
        inner=[validate_partition(tr[a],tr[b],tr,groups)
               for a,b in GroupKFold(inner_count).split(tr,groups=groups.iloc[tr])]
        splits.append((tr,te,inner))
    print("[AVISO] Plan interno histórico no disponible; CV interna espacial reconstruida. Folds externos intactos.",flush=True)
    return splits,{"inner_protocol":"reconstructed_with_current_sklearn","partition_sha256":None}
