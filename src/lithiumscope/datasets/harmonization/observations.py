"""Non-destructive source-scoped identity and replicate reconciliation."""
from __future__ import annotations

import hashlib
import json

import pandas as pd


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def reconcile_observations(frame, *, source, sample, replicate, payload):
    """Return positional decisions. Never resolve conflicting assays by row order.

    Only explicit replicate IDs permit differing measurements of a single
    physical sample. Unknown source/sample identity is not a deduplication key.
    Payload must contain coordinates, analytical semantics and measurements.
    """
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Reconciliation requires unique row and column identities")
    required = {source, sample, replicate, *payload}
    if not required <= set(frame) or not payload:
        raise ValueError("Reconciliation identity/payload fields absent")
    rows = []
    groups = {}
    for position, (_, row) in enumerate(frame.iterrows()):
        src, sid, rid = (_text(row[key]) for key in (source, sample, replicate))
        entry = {"position": position, "source": src, "sample": sid,
                 "replicate": rid, "decision": "retain", "reason": "unique_identity"}
        rows.append(entry)
        if not src or not sid:
            entry.update(reason="unresolved_identity_retained")
            continue
        groups.setdefault((src, sid), []).append(position)
    for positions in groups.values():
        if len(positions) == 1:
            continue
        # Unlabelled and explicitly labelled measurements cannot be equated.
        subgroups = {}
        for i in positions:
            subgroups.setdefault(rows[i]["replicate"], []).append(i)
        if "" in subgroups and len(subgroups) > 1:
            for i in positions:
                rows[i].update(decision="quarantine", reason="mixed_replicate_identity")
            continue
        for rid, indices in subgroups.items():
            distinct = frame.iloc[indices][list(payload)].drop_duplicates()
            if len(distinct) > 1:
                for i in indices:
                    rows[i].update(decision="quarantine", reason="conflicting_measurements")
            else:
                rows[indices[0]]["reason"] = "explicit_replicate" if rid else "identical_representative"
                for i in indices[1:]:
                    rows[i].update(decision="exclude", reason="identical_duplicate")
    return pd.DataFrame(rows, columns=["position", "source", "sample", "replicate", "decision", "reason"])


def stable_disambiguation(source, sample, replicate, row_identity=""):
    """Digest uses original identifiers, not lossy sanitized filenames."""
    data = json.dumps([str(source), str(sample), str(replicate), str(row_identity)],
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:24]
