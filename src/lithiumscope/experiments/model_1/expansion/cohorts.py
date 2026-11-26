"""Predeclared candidate review and fold-safe cohort membership; no estimator fits."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from lithiumscope.experiments.data_lab.candidate_screening import text
from lithiumscope.experiments.model_1.expansion.identity import multivariate_support

REVIEW_COLUMNS = ["candidate_id","raw_source_sha256","decision","method_evidence",
                  "crs_evidence","reference","notes"]
REVIEW_STATES = {"compatible","experimental_transfer","reject"}
POOLS = {"D1": {"convergent_review"}, "D2": {"high_transfer_review"},
         "D3": {"convergent_review","high_transfer_review"},
         "D4": {"convergent_review","high_transfer_review","other_domain_review"}}
GOOD_STATES = POOLS["D4"]


def review_candidates(audit, review=None):
    result = audit.copy()
    result["review_decision"] = "pending"
    result["review_reference"] = ""
    result["review_method_evidence"] = ""
    result["review_crs_evidence"] = ""
    result["review_notes"] = ""
    if review is not None:
        if not set(REVIEW_COLUMNS).issubset(review.columns):
            raise ValueError("Incomplete review ledger")
        review = review[REVIEW_COLUMNS].fillna("").astype(str)
        review = review.loc[review.decision.str.strip().ne("")]
        if review.candidate_id.duplicated().any() or not set(review.candidate_id).issubset(result.candidate_id):
            raise ValueError("Unknown or duplicate reviewed candidate")
        if not set(review.decision).issubset(REVIEW_STATES):
            raise ValueError("Unsupported review decision")
        idx = result.set_index("candidate_id",drop=False)
        for row in review.itertuples(index=False):
            current = idx.loc[row.candidate_id]
            if row.raw_source_sha256 != current.raw_source_sha256:
                raise ValueError("Review source hash differs")
            if row.decision != "reject":
                if (current.screening_state not in GOOD_STATES or not current.raw_verified or not current.prepared_coordinates_valid
                        or not all(str(v).strip() for v in (row.method_evidence,row.crs_evidence,row.reference))
                        or any(str(v).strip().casefold() in {"unknown","desconocido","n/a","nd"}
                               for v in (row.method_evidence,row.crs_evidence,row.reference))):
                    raise ValueError("Positive review requires verified source, screening and documented evidence")
                if row.decision == "compatible" and current.screening_state == "high_transfer_review":
                    raise ValueError("High intraplate candidates require explicit transfer review")
            mask = result.candidate_id.eq(row.candidate_id)
            result.loc[mask, ["review_decision","review_reference","review_method_evidence",
                              "review_crs_evidence","review_notes"]] = [
                row.decision,row.reference,row.method_evidence,row.crs_evidence,row.notes]
    result["experimental_eligible"] = (result.screening_state.isin(GOOD_STATES)
        & result.raw_verified & result.prepared_coordinates_valid & result.review_decision.isin({"compatible","experimental_transfer"}))
    result["approved_for_production"] = False
    return result


def _rank(value):
    return hashlib.sha256(("M1-expansion-v1|"+str(value)).encode()).hexdigest()


def reserve_holdout(audit, historical_groups, historical_publications):
    """Reserve whole candidate groups; quarantine every linked publication.

    All candidate data were already descriptively explored. This is a reserved
    development check, not an independently acquired external holdout.
    """
    result = audit.copy()
    pub = text(result,"source_publication")
    eligible = result.experimental_eligible
    possible = eligible & ~result.spatial_group.isin(historical_groups) & pub.ne("") & ~pub.isin(historical_publications)
    groups = sorted(set(result.loc[possible,"spatial_group"]),key=_rank)
    selected = set(groups[:max(1,len(groups)//5)]) if len(groups)>=5 else set()
    held = possible & result.spatial_group.isin(selected)
    held_publications = set(pub[held])
    quarantine = eligible & (result.spatial_group.isin(selected) | pub.isin(held_publications)) & ~held
    result["candidate_partition"] = np.select([held,quarantine,eligible],
        ["reserved_development_check","quarantine_linked_to_reserve","train_pool"],default="review_only")
    return result


def train_domain_fraction(candidate_x, train_x, numeric):
    # No TEST features or labels are passed here; zero available => ineligible.
    b=train_x[numeric].apply(pd.to_numeric,errors="coerce")
    c=candidate_x[numeric].apply(pd.to_numeric,errors="coerce")
    low,high=b.quantile(.01),b.quantile(.99)
    available=c.notna() & low.notna() & high.notna()
    outside=(c.lt(low)|c.gt(high)) & available
    return outside.sum(axis=1).div(available.sum(axis=1).replace(0,np.nan))


def fold_membership(audit, candidate_x, base_x, folds, historical_publications, numeric):
    """Protect every original TEST; the extension is TRAIN-only.

    D4 other-domain selection uses only features of the relevant historical TRAIN.
    Publication conflicts are blocked when the historical source records a publication.
    """
    if len(base_x)!=len(folds) or len(historical_publications)!=len(folds):
        raise ValueError("Historical features/folds/publications misaligned")
    if not candidate_x.index.equals(audit.index):
        raise ValueError("Candidate feature identity differs")
    records=[]
    pub=text(audit,"source_publication")
    for fold in sorted(folds.outer_fold.unique()):
        test=folds.outer_fold.eq(fold).to_numpy()
        blocked_groups=set(folds.loc[test,"spatial_group"])
        blocked_publications=set(historical_publications.iloc[np.flatnonzero(test)])-{""}
        out=train_domain_fraction(candidate_x,base_x.iloc[np.flatnonzero(~test)],numeric)
        safe=(audit.candidate_partition.eq("train_pool") & ~audit.spatial_group.isin(blocked_groups)
              & ~pub.isin(blocked_publications))
        other_pool=safe & audit.screening_state.eq("other_domain_review") & audit.review_decision.eq("compatible")
        chemistry_ok=pd.Series(False,index=audit.index)
        if other_pool.any():
            support=multivariate_support(candidate_x.loc[other_pool],base_x.iloc[np.flatnonzero(~test)],numeric)
            chemistry_ok.loc[other_pool]=support.chemistry_nn_supported
        for cohort,pools in POOLS.items():
            allowed=safe & audit.screening_state.isin(pools)
            if cohort=="D4":
                other=audit.screening_state.eq("other_domain_review")
                allowed &= ~other | (out.le(.25) & chemistry_ok & audit.review_decision.eq("compatible"))
            records.append(pd.DataFrame({"fold":int(fold),"cohort":cohort,
                "candidate_id":audit.candidate_id,"train_domain_fraction":out,
                "other_domain_chemistry_supported":chemistry_ok,
                "candidate_train":allowed}))
    return pd.concat(records,ignore_index=True)
