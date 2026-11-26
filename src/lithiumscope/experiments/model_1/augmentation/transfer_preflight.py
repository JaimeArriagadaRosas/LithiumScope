"""Conservative directed transfer preflight; no fitting or independent-holdout claim."""
from datetime import datetime, timezone
from uuid import uuid4
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.management.local_sources import local_training_source
from lithiumscope.experiments.data_lab.candidate_screening import CHEM, num
from lithiumscope.experiments.model_1.balanced.academic import cohort
from lithiumscope.experiments.model_1.expansion.academic_bridge import PILOT, SEED, check_pilot
from lithiumscope.experiments.model_1.expansion.identity import multivariate_support
from lithiumscope.experiments.model_1.expansion.cohorts import train_domain_fraction
from lithiumscope.experiments.model_1.expansion.sensitivity_data import read
from lithiumscope.experiments.model_1.augmentation.chen_profile import ROOT,SOURCE,NAME,SHA
from lithiumscope.experiments.model_1.augmentation.chen_overlap import ID, publication_sets
from lithiumscope.experiments.model_1.augmentation.whole_rock_audit import PUNA,MAC,FILES,META
from lithiumscope.results.atomic_exports import write_table,write_legacy_evidence_json

CHEN_RUN="20261126T210740_ed33aa31"
CHEN_MANIFEST="b0bfff9a1ee3e7660a5e7388acd2b3ce1886ca6acc1fe6fc7895f057eca44925"
WHOLE_RUN="20261126T205508_dccd2ea0"
MAX_SPAN=.1
QUARANTINED_IDS={"17558"}
YEAR=re.compile(r"\b(?:18|19|20)\d{2}\b")
AUTHOR=re.compile(r"^\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]+)")


def author_year(value):
    """Conservative first-author/year alias; not bibliographic identity proof."""
    text=str(value).strip()
    a=AUTHOR.match(text)
    if not a:
        return set()
    # Prefer year in brackets, as in GEOROC bibliography.
    years=re.findall(r"\[((?:18|19|20)\d{2})\]",text) or YEAR.findall(text)
    return {a[1].casefold()+":"+v for v in years}


def verified_run(folder,scope,expected=None):
    path=folder/"manifest.json"
    if expected is not None and file_sha256(path)!=expected:
        raise ValueError("Audit manifest differs")
    manifest=read(path)
    if (manifest.get("scope")!=scope or manifest.get("training") is not False
            or manifest.get("new_rows_approved")!=0 or manifest.get("state")!="review_required"):
        raise ValueError("Unexpected audit state")
    for name,digest in manifest["outputs"].items():
        from pathlib import Path
        if Path(name).name!=name or file_sha256(folder/name)!=digest:
            raise ValueError("Audit output differs: "+name)
    return manifest,file_sha256(path)


def geographic_ok(frame):
    xy=num(frame,["Latitude_min","Latitude_max","Longitude_min","Longitude_max"])
    lat=xy.Latitude_max-xy.Latitude_min
    lon=xy.Longitude_max-xy.Longitude_min
    return (xy.Latitude_min.between(-90,90)&xy.Latitude_max.between(-90,90)&
        xy.Longitude_min.between(-180,180)&xy.Longitude_max.between(-180,180)&
        lat.between(0,MAX_SPAN)&lon.between(0,MAX_SPAN))


def static_gate(rows,reference_index,historical_aliases):
    result=rows.copy()
    _,pubs=publication_sets(rows.source_publication,reference_index)
    ids=[set(ID.findall(str(v))) for v in rows.source_publication]
    aliases=[set().union(*(author_year(t) for i in rr for t in reference_index.get(i,set()))) for rr in ids]
    result["geographic_precision_ok"]=geographic_ok(rows)
    result["obsidian_publication_review"]=[bool(v&QUARANTINED_IDS) for v in ids]
    result["historical_author_year_overlap"]=[bool(v&historical_aliases) for v in aliases]
    result["static_review_candidate"]=(result.transfer_review_candidate.eq(True)&
        result.geographic_precision_ok&~result.obsidian_publication_review&~result.historical_author_year_overlap)
    return result,aliases,pubs


def fold_gate(rows,train_x,test_groups,chemistry,*,test_aliases,aliases):
    """Use only TRAIN chemistry to estimate support; block known TEST groups."""
    support=multivariate_support(rows,train_x,chemistry)
    fraction=train_domain_fraction(rows,train_x,chemistry)
    blocked=rows.spatial_group.isin(test_groups)
    pub_block=pd.Series([bool(v&test_aliases) for v in aliases],index=rows.index)
    eligible=(rows.static_review_candidate.eq(True)&~blocked&~pub_block&
              support.chemistry_nn_supported&fraction.le(.25))
    return pd.DataFrame({"candidate_id":rows.candidate_id,"source_dataset":rows.source_dataset,
        "Li_icpms":rows.Li_icpms,"static_review_candidate":rows.static_review_candidate,
        "test_group_overlap":blocked,"test_publication_alias_overlap":pub_block,
        "chemistry_nn_distance":support.chemistry_nn_distance,
        "chemistry_nn_q95":support.chemistry_nn_q95,"chemistry_supported":support.chemistry_nn_supported,
        "marginal_ood_fraction":fraction,"fold_transfer_review_candidate":eligible})


def summarize_plans(parts):
    """Build fold counts with explicit column access (tail is a DataFrame method)."""
    plans=pd.concat(parts,ignore_index=True)
    plans["tail"]=plans["Li_icpms"].gt(40)
    plans["accepted_tail"]=plans["tail"] & plans["fold_transfer_review_candidate"]
    summary=plans.groupby(["fold","source_dataset"])[
        ["static_review_candidate","fold_transfer_review_candidate","accepted_tail"]].sum().reset_index()
    return plans,summary


def run():
    chen=RESULTS_DIR/"generalization/model_1_chen_overlap_audit"/CHEN_RUN
    whole=RESULTS_DIR/"generalization/model_1_directed_whole_rock_audit"/WHOLE_RUN
    cm,ch=verified_run(chen,"chen_overlap_screen",CHEN_MANIFEST)
    # The whole-rock manifest lacks a scope field in its original implementation.
    wm=read(whole/"manifest.json")
    if (wm.get("state")!="review_required" or wm.get("training") is not False
            or wm.get("new_rows_approved")!=0 or wm.get("rows")!=35
            or wm.get("canonical_columns_are_review_aliases") is not True):
        raise ValueError("Whole-rock audit identity differs")
    for name,digest in wm["outputs"].items():
        from pathlib import Path
        if Path(name).name!=name or file_sha256(whole/name)!=digest:
            raise ValueError("Whole-rock evidence differs")
    wh=file_sha256(whole/"manifest.json")
    for key,(name,digest) in FILES.items():
        if file_sha256(ROOT/key/name)!=digest:
            raise ValueError("Whole-rock original differs")
    if file_sha256(ROOT/PUNA/META[0])!=META[1] or file_sha256(ROOT/SOURCE/NAME)!=SHA:
        raise ValueError("Original differs")
    rows=pd.read_csv(chen/"row_review.csv")
    if len(rows)!=3680 or not rows.excel_row.is_unique or rows.transfer_review_candidate.sum()!=2303:
        raise ValueError("Chen candidate inventory differs")
    bibliography=pd.read_csv(chen/"bibliography.csv",dtype={"reference_id":str})
    index={k:set(v.reference) for k,v in bibliography.groupby("reference_id")}
    source=local_training_source("model_1")
    digest=file_sha256(source)
    if digest!=cm["source_sha256"] or digest!=wm["source_sha256"]:
        raise ValueError("Historical source differs")
    prepared,keep,domain=cohort(source)
    x,y,groups=(v.loc[keep] for v in (prepared.x,prepared.y,prepared.groups))
    if len(x)!=3263:
        raise ValueError("Frozen cohort differs")
    historical=prepared.raw_frame
    refcols=[c for c in ("source_publication","Geochemistry Reference") if c in historical]
    history_aliases={i:set().union(*(author_year(v) for v in row)) for i,row in historical[refcols].iterrows()}
    known=set().union(*history_aliases.values())
    rows,aliases,pubs=static_gate(rows,index,known)
    rows["candidate_id"]="chen:"+rows.excel_row.astype(str)
    rows["source_dataset"]=SOURCE
    wr=pd.read_csv(whole/"row_review.csv")
    puna=wr.loc[wr.source_dataset.eq(PUNA)].copy()
    if len(puna)!=6 or int(puna.academic_review_candidate.sum())!=6 or len(wr)!=35:
        raise ValueError("Puna inventory differs")
    puna["candidate_id"]="puna:"+puna.source_sample.astype(str)
    # DOI metadata and method documented, CRS/publication equivalence still pending.
    puna["static_review_candidate"]=puna.academic_review_candidate.eq(True)
    puna["obsidian_publication_review"]=False
    puna["historical_author_year_overlap"]=False
    puna["geographic_precision_ok"]=True
    # Cross-source near-duplicate comparison using target-independent location/chemistry.
    from lithiumscope.experiments.model_1.augmentation.chen_overlap import overlap_edges
    puna["excel_row"]=puna.analysis_excel_row
    cross=overlap_edges(puna,rows,[set() for _ in range(len(puna))],pubs,"chen")
    blocked={v["excel_row"] for v in cross}
    puna.loc[puna.excel_row.isin(blocked),"static_review_candidate"]=False
    chen_positions={v["comparison_position"] for v in cross}
    rows.loc[rows.index.astype(str).isin(chen_positions),"static_review_candidate"]=False
    joined=pd.concat([rows,puna],ignore_index=True,sort=False)
    aliases=aliases+[{"fernandez:2020","fernandez-turiel:2020"} for _ in range(len(puna))]
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(x))
    check_pilot(RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT,x,y,groups,splits,digest,domain)
    chemistry=[c for c in CHEM if c in x]
    plans=[]
    for fold,(tr,te) in enumerate(splits,1):
        test_aliases=set().union(*(history_aliases.get(i,set()) for i in x.index[te]))
        flags=fold_gate(joined,x.iloc[tr],set(groups.iloc[te]),chemistry,
            test_aliases=test_aliases,aliases=aliases)
        flags.insert(0,"fold",fold)
        plans.append(flags)
    plans,summary=summarize_plans(plans)
    # No independent reserve is certified from the previously explored compilation.
    out=RESULTS_DIR/"generalization/model_1_directed_transfer_preflight"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    columns=["candidate_id","source_dataset","source_publication","Li_icpms","spatial_group",
        "static_review_candidate","geographic_precision_ok","obsidian_publication_review","historical_author_year_overlap"]
    tables={"static_review.csv":joined[columns],"fold_review.csv":plans,"summary.csv":summary,
        "cross_source_edges.csv":pd.DataFrame(cross,columns=["excel_row","comparison","comparison_position",
            "comparison_excel_row","site_overlap","publication_overlap","shared_chemistry","shared_major_oxides"])}
    for name,frame in tables.items():
        write_table(out/name,frame)
    write_legacy_evidence_json(out/"manifest.json",{"state":"review_required","scope":"directed_transfer_preflight",
        "source_sha256":digest,"chen_manifest_sha256":ch,"whole_rock_manifest_sha256":wh,
        "max_coordinate_span_degrees":MAX_SPAN,"quarantined_reference_ids":sorted(QUARANTINED_IDS),
        "target_used_for_identity":False,"fold_domain_fit_train_only":True,"new_rows_approved":0,
        "independent_holdout_certified":False,"training":False,
        "outputs":{name:file_sha256(out/name) for name in tables}})
    print(summary.to_string(index=False))
    print(f"Estado: review_required | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    run()
