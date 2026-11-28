"""Compare current and archived GEOROC views without re-running their campaigns."""
from collections import Counter
import re
import numpy as np
import pandas as pd

from ..screening.candidates import sample, num, CHEM, group
from ..source_studies.historical_evidence import HISTORICAL, verify_manifest
from ..source_studies.georoc.candidate_eda import overlap_flags
from lithiumscope.core.scientific.chemical_support import chemical_support
from .evidence import BASE_SHA, read
from lithiumscope.experiments.model_1.data_expansion.sensitivity_data import tokens, historical_tokens

KEYS = ("georoc_andean_202609", "georoc_felsic_rocks")


def view(profile):
    return pd.read_csv(profile["path"] / "tables/candidate_view.csv", low_memory=False)


def keys(frame):
    """Comparison key only; includes reported Li, never used for recovery or inference."""
    values = num(frame, ["Latitude", "Longitude", "Li_icpms"])
    name = sample(frame)
    valid = values.notna().all(axis=1) & name.ne("")
    result = (frame.source_file.astype(str) + "|" + name + "|" +
              values.Latitude.round(5).astype(str) + "|" + values.Longitude.round(5).astype(str) +
              "|" + values.Li_icpms.round(6).astype(str))
    return result.where(valid, "")


def reconcile(profiles, base, root, lower, upper):
    relative, digest = HISTORICAL[KEYS[0]][0]
    folder = root / relative
    manifest = verify_manifest(folder, digest)
    if manifest.get("source_sha256") != BASE_SHA:
        raise ValueError("Archived expansion belongs to a different source")
    old = pd.read_csv(folder / "row_audit.csv", low_memory=False)
    if not old.candidate_id.is_unique:
        raise ValueError("Archived candidate identity duplicated")
    counts, review, current = [], [], []
    for key in KEYS:
        if key not in profiles:
            continue
        new = view(profiles[key])
        before = old.loc[old.source_dataset.eq(key)].copy()
        a, b = keys(before), keys(new)
        old_counts, new_counts = Counter(a[a.ne("")]), Counter(b[b.ne("")])
        lookup = before.assign(comparison_key=a)
        unique = lookup.loc[lookup.comparison_key.ne("") & ~lookup.comparison_key.duplicated(keep=False)]
        unique = unique.set_index("comparison_key")
        qc = num(new, ["SUM (no water)"]).iloc[:, 0].between(lower, upper)
        flags = overlap_flags(new, base)
        no_flag = ~flags[["source_unique_id_overlap", "coordinate_li_overlap",
                         "within_candidate_coordinate_li"]].any(axis=1)
        y = num(new, ["Li_icpms"]).iloc[:,0]
        selected = qc & no_flag & (y.gt(696) if key == KEYS[1] else True)
        expected=profiles[key]["manifest"]["summary"]
        expected_count=expected["above_696_qc_no_identity_flag" if key==KEYS[1] else "qc_no_identity_flag_rows"]
        if len(new)!=expected["candidate_rows"] or int(selected.sum())!=expected_count:
            raise ValueError("Current candidate screening differs from verified EDA: "+key)
        detail = new.loc[selected].copy()
        detail["comparison_key"] = b.loc[selected]
        detail["archived_candidate_id"] = detail.comparison_key.map(unique.candidate_id)
        detail["archived_screening_state"] = detail.comparison_key.map(unique.screening_state)
        detail["unique_comparison_key"] = detail.comparison_key.map(old_counts).eq(1) & detail.comparison_key.map(new_counts).eq(1)
        detail["review_state"] = np.where(detail.unique_comparison_key,
            "historical_key_reused_not_independence", "membership_unresolved")
        structures=read(profiles[key]["path"]/"tables/raw_structure_audit.json")
        bibliography={}
        for file,structure in structures.items():
            for row in structure.get("footer",[]):
                if not str(row["section"]).strip().lower().startswith(("references","bibliography","citations")):
                    continue
                for cell in row["cells"]:
                    match=re.match(r"^\s*\[(\d+)\]\s*(.+)",cell)
                    if match:
                        bibliography.setdefault((file,match[1]),set()).add(match[2].strip())
        references=[]
        for item in detail.itertuples(index=False):
            ids=re.findall(r"\[(\d+)\]",str(item.source_publication))
            resolved=[bibliography.get((item.source_file,i),set()) for i in ids]
            references.append(" | ".join(next(iter(v)) for v in resolved)
                if ids and all(len(v)==1 for v in resolved) else "")
        detail["resolved_publications"]=references
        detail["approved_for_training"] = False
        detail["review_id"] = key + "|" + detail.source_file.astype(str) + "|" + detail.source_row_position.astype(str)
        current.append(detail)
        for k in sorted(set(old_counts) | set(new_counts)):
            if old_counts[k] != new_counts[k]:
                review.append({"source":key,"comparison_key":k,"archived_rows":old_counts[k],
                               "current_rows":new_counts[k],"cause":"unresolved_filter_or_identity_difference"})
        counts.append({"source":key,"archived_rows":len(before),"current_rows":len(new),
            "matched_key_occurrences":sum((old_counts & new_counts).values()),
            "archived_unkeyed":int(a.eq("").sum()),"current_unkeyed":int(b.eq("").sum()),
            "review_rows":len(detail),"review_unique_archived_keys":int(detail.unique_comparison_key.sum()),
            "approved_rows":0})
    return pd.DataFrame(counts), pd.DataFrame(review, columns=["source","comparison_key",
        "archived_rows","current_rows","cause"]), pd.concat(current, ignore_index=True) if current else pd.DataFrame()


def candidate_preflight(candidates, x, groups, plan, *, historical=None):
    """Diagnostic support computed only from each TRAIN, never admit rows."""
    columns = ["review_id","outer_fold","inner_fold","chemistry_nn_supported","group_reserved",
               "method_known","crs_known","approved_for_training"]
    rows = []
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    cx = candidates.reindex(columns=CHEM)
    concentrations = num(candidates,["Li_icpms"]).Li_icpms
    sources = candidates.source_dataset
    old_refs=historical_tokens(historical) if historical is not None else [set() for _ in range(len(x))]
    if len(old_refs)!=len(x):
        raise ValueError("Historical publication identities not aligned")
    candidate_refs=[tokens(a)|tokens(b) for a,b in zip(candidates.source_publication,candidates.resolved_publications,strict=True)]
    cg = group(candidates)
    for fold,(train,test,inner) in enumerate(plan,1):
        for i,(fit,reserved) in enumerate([(train,test),*inner]):
            support = chemical_support(cx, x.iloc[fit], [c for c in CHEM if c in x])
            table = support.reset_index(drop=True)
            table["review_id"] = candidates.review_id.to_numpy()
            table["source"] = sources.to_numpy()
            table["candidate_li_ppm"] = concentrations.to_numpy()
            table["above_696"] = concentrations.gt(696).to_numpy()
            table["outer_fold"],table["inner_fold"] = fold,i
            # No reserved labels are used; group exclusion is a prerequisite only.
            # Inner folds must also exclude the outer TEST groups.
            blocked = set(groups.iloc[reserved]) | set(groups.iloc[test])
            table["group_reserved"] = cg.isin(blocked).to_numpy()
            reserved_positions=set(map(int,reserved)) | set(map(int,test))
            reserved_refs=set().union(*(old_refs[j] for j in reserved_positions))
            table["known_reserved_publication_overlap"]=[bool(v&reserved_refs) for v in candidate_refs]
            table["reserved_rows_without_publication"]=sum(not old_refs[j] for j in reserved_positions)
            table["candidate_publication_resolved"]=candidates.resolved_publications.ne("").to_numpy()
            table["publication_independence_certified"]=False
            for name,column in (("method_known","source_li_method"),("crs_known","source_coordinate_crs")):
                value = candidates.get(column,pd.Series("",index=candidates.index)).fillna("").astype(str).str.strip().str.lower()
                table[name] = ~value.isin(["","unknown","nan","none","<na>","unconfirmed"]) & ~value.str.startswith("review_")
            table["approved_for_training"] = False
            rows.append(table)
    return pd.concat(rows, ignore_index=True)
