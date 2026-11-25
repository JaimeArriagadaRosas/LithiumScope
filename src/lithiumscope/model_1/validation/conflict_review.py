"""Documentary source-conflict review, bound to frozen source bytes and row identity."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.pipeline_audit import stable_source_row_id
from lithiumscope.model_1.validation.observation_audit import audit_prepared_observations

DECISIONS={"retain_verified_replicate","retain_distinct_sample","exclude_confirmed_duplicate",
           "exclude_confirmed_incompatible_measurement","unresolved"}
EXCLUDE={"exclude_confirmed_duplicate","exclude_confirmed_incompatible_measurement"}


def review_register(prepared):
    audit,_=audit_prepared_observations(prepared)
    audit["source_row_id"]=[stable_source_row_id(prepared.audit.dataset_sha256,i) for i in prepared.frame.index]
    audit["dataset_sha256"]=prepared.audit.dataset_sha256
    # Document conflicting observations, exact duplicates, unresolved IDs and shared sites.
    needs=(audit.reason.ne("unique_identity") | audit.cross_source_coordinate_match)
    result=audit.loc[needs].copy()
    result["review_decision"]="unresolved"
    for name in ("evidence_path","evidence_sha256","evidence_locator","reviewer","rationale","retained_source_row_id"):
        result[name]=""
    return result


def apply_review(prepared, ledger, *, evidence_root):
    register=review_register(prepared)
    by_id=register.set_index("source_row_id")
    needed={"source_row_id","dataset_sha256","review_decision","evidence_path","evidence_sha256",
            "evidence_locator","reviewer","rationale","retained_source_row_id"}
    if not needed<=set(ledger) or ledger.source_row_id.duplicated().any():
        raise ValueError("Missing or duplicated documentary review identity")
    if not set(ledger.source_row_id)<=set(by_id.index):
        raise ValueError("Ledger row not in review candidates")
    if not ledger.dataset_sha256.eq(prepared.audit.dataset_sha256).all():
        raise ValueError("Ledger belongs to different source bytes")
    all_ids={stable_source_row_id(prepared.audit.dataset_sha256,i):p for p,i in enumerate(prepared.frame.index)}
    root=Path(evidence_root).resolve()
    mask=np.ones(len(prepared.y),dtype=bool)
    verified=[]
    def nonempty(value):
        return not pd.isna(value) and bool(str(value).strip())
    for row in ledger.itertuples(index=False):
        decision=row.review_decision
        if decision not in DECISIONS:
            raise ValueError("Unknown documentary decision")
        if decision=="unresolved":
            continue
        if not all(nonempty(getattr(row,key)) for key in ("reviewer","rationale","evidence_locator","evidence_path","evidence_sha256")):
            raise ValueError("Reviewed decision requires documentary evidence and attribution")
        evidence=(root/str(row.evidence_path)).resolve()
        if not evidence.is_relative_to(root) or not evidence.is_file() or file_sha256(evidence)!=row.evidence_sha256:
            raise ValueError("Evidence missing, outside root, or changed")
        p=all_ids[row.source_row_id]
        if decision=="exclude_confirmed_duplicate":
            if row.retained_source_row_id not in all_ids or row.retained_source_row_id==row.source_row_id:
                raise ValueError("Confirmed duplicate requires distinct retained row")
        if decision in EXCLUDE:
            mask[p]=False
        verified.append({"source_row_id":row.source_row_id,"position":p,"review_decision":decision,
                         "document_sha256":row.evidence_sha256,"document_locator":row.evidence_locator,
                         "reviewer":row.reviewer,"rationale":row.rationale,
                         "retained_source_row_id":row.retained_source_row_id if nonempty(row.retained_source_row_id) else ""})
    for decision in verified:
        retained=decision["retained_source_row_id"]
        if decision["review_decision"]=="exclude_confirmed_duplicate" and not mask[all_ids[retained]]:
            raise ValueError("Duplicate counterpart is also excluded")
    if not mask.any():
        raise ValueError("Review cannot exclude entire cohort")
    return mask, {"review_candidates":len(register),"documented_decisions":len(verified),
                  "unresolved":len(register)-len(verified),"train_exclusions":int((~mask).sum()),
                  "decisions":verified,"historical_dataset_mutated":False,
                  "review_content_certified":False}


def conflict_arms(prepared, ledger, algorithms, *, evidence_root, seed):
    from lithiumscope.experiments.shared.paired_suite import Arm
    from lithiumscope.core.reproducibility import canonical_json_hash
    mask,audit=apply_review(prepared,ledger,evidence_root=evidence_root)
    if mask.all():
        raise ValueError("No documented exclusions for controlled comparison")
    policy={"seed":seed,"review_sha256":canonical_json_hash(audit),
            "test_policy":"all_original_test_rows_retained","review_independence_certified":False}
    arms=[]
    for algorithm in algorithms:
        name=f"{algorithm}_all_original_train"
        arms.extend([Arm(name,algorithm,None,prepared.x,{**policy,"cohort":"all_train"}),
                     Arm(f"{algorithm}_documented_exclusions",algorithm,name,prepared.x,
                         {**policy,"cohort":"train_only_reviewed_exclusions"},mask)])
    return arms,audit
