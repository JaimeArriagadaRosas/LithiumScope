"""Deterministic descriptive top-1 evaluation for three independent candidate sites.

Three locations come from ONE geographically held-out fold per draw.
Ranks M2 scores against observed median Li per site, without tuning labels.
The same triplets are evaluated for all arms. Repeated draws do not count as
independent external validation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


REQUIRED={"arm","fold","test_index","site_screening_id","sample_id",
          "li_ppm","score","y_true"}


def ranking_evaluation(oof:pd.DataFrame, *,triples_per_fold:int=250,
                       seed:int=42)->tuple[dict,pd.DataFrame]:
    if triples_per_fold<1 or oof.empty or not REQUIRED.issubset(oof):
        raise ValueError("Incomplete three-zone ranking evaluation input")
    arms=sorted(oof.arm.astype(str).unique())
    first=None
    units={}
    for arm in arms:
        df=oof[oof.arm==arm].sort_values("test_index").reset_index(drop=True)
        if df.test_index.duplicated().any():
            raise ValueError("Duplicate test sample predictions")
        if first is None:
            first=df[["test_index","fold","sample_id","site_screening_id",
                      "li_ppm","y_true"]].copy()
        else:
            columns=list(first.columns)
            check=df[columns]
            if len(check)!=len(first) or not check.reset_index(drop=True).equals(first):
                raise ValueError("All ranking arms must share exactly the same held-out rows")
        if df.groupby("site_screening_id").fold.nunique().max()>1:
            raise ValueError("A candidate site occurs in multiple held-out folds")
        summary=df.groupby(["fold","site_screening_id"],sort=True).agg(
            li_ppm=("li_ppm","median"),score=("score","median"),
            positive=("y_true","max"),samples=("sample_id","size")
        ).reset_index()
        units[arm]=summary
    baseline=units[arms[0]]
    # Index once per competitor rather than rebuilding index for every triplet.
    indexed_units={
        name:frame.set_index("site_screening_id")
        for name,frame in units.items()
    }
    rng=np.random.default_rng(seed)
    cases=[]
    for fold,sites in baseline.groupby("fold",sort=True):
        site_ids=sites.site_screening_id.astype(str).tolist()
        if len(site_ids)<3:raise ValueError("Fewer than three distinct test sites")
        for draw in range(triples_per_fold):
            selected=rng.choice(site_ids,3,replace=False).tolist()
            scores={}
            values=[]
            positive=[]
            for arm in arms:
                sample=indexed_units[arm].loc[selected]
                v=sample.li_ppm.to_numpy(float)
                s=sample.score.to_numpy(float)
                if not np.isfinite(s).all():
                    raise ValueError("Nonfinite ranking score")
                if not len(values):
                    values=v.tolist()
                    positive=sample.positive.to_numpy(int).tolist()
                elif not np.allclose(v,values,atol=0,rtol=0):
                    raise ValueError("Geochemical site references changed")
                scores[arm]=s
            best_indices=np.flatnonzero(
                np.isclose(values,max(values),rtol=0,atol=1e-9))
            if len(best_indices)!=1:continue
            true_best=int(best_indices[0])
            for arm in arms:
                sc=scores[arm]
                top=np.flatnonzero(np.isclose(sc,max(sc),rtol=0,atol=1e-12))
                chose=int(top[0]) if len(top)==1 else -1
                cases.append({
                    "arm":arm,"fold":int(fold),"draw":draw,
                    "candidate_site_1":selected[0],
                    "candidate_site_2":selected[1],
                    "candidate_site_3":selected[2],
                    "li_1_ppm":float(values[0]),"li_2_ppm":float(values[1]),
                    "li_3_ppm":float(values[2]),
                    "score_1":float(sc[0]),"score_2":float(sc[1]),
                    "score_3":float(sc[2]),
                    "chosen_site":selected[chose] if chose>=0 else "",
                    "true_best_site":selected[true_best],
                    "top1_match":bool(chose==true_best),
                    "positive_in_triplet":bool(any(positive)),
                    "all_three_distinct_sites":True,
                    "score_tie_abstention":bool(chose==-1),
                })
    evidence=pd.DataFrame(cases)
    if evidence.empty:
        raise ValueError("No unambiguous three-site cases available")
    totals={}
    first_sets=None
    for arm,g in evidence.groupby("arm"):
        keys=set(zip(g.fold,g.draw))
        if first_sets is None:first_sets=keys
        elif keys!=first_sets:raise ValueError("Ranking comparisons used different draws")
        positives=g[g.positive_in_triplet]
        totals[arm]={
            "triple_cases":len(g),
            "folds":int(g.fold.nunique()),
            "top1_matches":int(g.top1_match.sum()),
            "top1_accuracy":float(g.top1_match.mean()),
            "cases_containing_a_high_Li_site":len(positives),
            "top1_accuracy_when_high_Li_present":(
                float(positives.top1_match.mean()) if len(positives) else None),
            "score_tie_abstentions":int(g.score_tie_abstention.sum()),
        }
    return {
        "status":"descriptive_historical_oof_only",
        "triples_per_fold_requested":int(triples_per_fold),
        "seed":int(seed),
        "distinct_sites_per_case":3,
        "candidate_sets_shared_across_arms":True,
        "chosen_target":"highest observed site-median Li among three",
        "not_an_external_validation":True,
        "random_choice_reference_accuracy":1/3,
        "arms":totals,
    },evidence



def rank_three_candidates(
    zones:dict, *, bundle:dict,
    model1_lithium_ppm:dict|None=None
)->dict:
    """Rank three DISTINCT input images using one candidate M2.

    Model 1 ppm are shown as complementary evidence, never blended into M2.
    Refuses to declare one best among all 3 if any input abstains. An active
    release bundle without explicit opt-in metadata is not accepted here.
    """
    from pathlib import Path
    from lithiumscope.core.hashing import file_sha256
    from lithiumscope.model_2.prediction.predictor import score_model_2_image

    if len(zones)!=3 or any(not str(k).strip() for k in zones):
        raise ValueError("Three named candidate zones are required")
    if bundle.get("experimental_only") is not True:
        raise ValueError("Three-site review requires an explicit experimental candidate")
    if float(bundle.get("quality_min_joint_valid",0))<=0:
        raise ValueError("Candidate lacks quality-abstention contract")
    inputs=[Path(v) for v in zones.values()]
    if not all(p.is_file() for p in inputs):
        raise FileNotFoundError("One or more candidate imagery files are missing")
    fingerprints=[file_sha256(p) for p in inputs]
    if len(set(fingerprints))!=3:
        raise ValueError("Three distinct candidate images required; repeated patch detected")
    measured=[]
    for name,path in zones.items():
        prediction,diagnostics=score_model_2_image(Path(path),bundle=bundle)
        ppm=None
        if model1_lithium_ppm and name in model1_lithium_ppm:
            ppm=float(model1_lithium_ppm[name])
            if not np.isfinite(ppm) or ppm<0:
                raise ValueError("Invalid supplementary Model 1 Li estimate")
        measured.append({
            "zone":str(name),
            "model_1_predicted_li_ppm":ppm,
            "model_2_spectral_score":prediction.get("prospectivity_score"),
            "image_quality_status":prediction.get("quality_status","UNKNOWN"),
            "warning":prediction.get("warning"),
        })
    if any(x["model_2_spectral_score"] is None for x in measured):
        return {
            "status":"ABSTAIN_ONE_OR_MORE_ZONES",
            "three_zones_comparable":False,
            "best_zone":None,
            "reason":"At least one input failed the new candidate quality gate",
            "zones":measured,
        }
    ranked=sorted(measured,key=lambda d:d["model_2_spectral_score"],reverse=True)
    for rank,entry in enumerate(ranked,1):
        entry["spectral_rank"]=rank
    return {
        "status":"RANKED_EXPERIMENTALLY",
        "three_zones_comparable":True,
        "best_zone":ranked[0]["zone"],
        "ranking_criterion":"M2 descending spectral score",
        "model1_ppm_not_used_to_fit_or_rank":True,
        "not_an_extraction_feasibility_assessment":True,
        "zones":ranked,
    }
