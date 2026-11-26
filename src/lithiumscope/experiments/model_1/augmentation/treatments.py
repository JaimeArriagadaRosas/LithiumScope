"""TRAIN-only bounded enrichment, with explicit parent identities and no invented sites."""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from lithiumscope.experiments.data_lab.candidate_screening import CHEM
from lithiumscope.experiments.model_1.expansion.academic_features import measurement_blocks
from lithiumscope.experiments.model_1.targeted.policies import weights
from lithiumscope.model_1.preparation.feature_engineering import add_geochemical_features
from lithiumscope.model_1.training.preprocessing.tabular import OXIDES, DERIVED

ARMS=("control","rare_copies","density_kde","relevance_copies","smoter_enriched","gaussian_enriched")
TAIL=40.
MAX_PARENT=4
MAX_PAIR=2


def validate(x,y,groups,sources):
    if (not x.index.is_unique or not all(x.index.equals(v.index) for v in (y,groups,sources))
            or not len(x) or groups.isna().any() or sources.isna().any()
            or not np.isfinite(y).all() or y.lt(0).any()):
        raise ValueError("Invalid augmentation TRAIN identities or labels")


def copies(y,groups,rng,mode):
    rare=np.flatnonzero(y.to_numpy(float)>TAIL)
    budget=2*len(rare)
    if mode=="rare_copies":
        return np.repeat(rare,2)
    if not budget:
        return np.array([],dtype=int)
    # WERCS-inspired bounded relevance oversampling only, not original WERCS.
    q=float(y.quantile(.90))
    relevance=np.clip((y.to_numpy(float)-q)/max(TAIL-q,1.),0.,1.)
    eligible=np.flatnonzero(relevance>0)
    if not len(eligible):
        return np.array([],dtype=int)
    pool=np.repeat(eligible,MAX_PARENT)
    freq=groups.iloc[eligible].value_counts()
    prob=np.repeat(relevance[eligible]/groups.iloc[eligible].map(freq).to_numpy(float),MAX_PARENT)
    return rng.choice(pool,min(budget,len(pool)),replace=False,p=prob/prob.sum())


def compatible_pairs(x,y,groups,sources,canonical,measured,categorical):
    # Same exact site, source, categories and canonical missingness.
    raw=x[measured].to_numpy(float)
    chem=x[canonical].to_numpy(float)
    coord=x[["Latitude","Longitude"]].to_numpy(float)
    if not np.isfinite(coord).all() or np.isinf(raw).any() or (raw[np.isfinite(raw)]<0).any():
        raise ValueError("Invalid physical predictors")
    cats=x[categorical].astype("string").fillna("<missing>").to_numpy()
    log=np.log1p(chem)
    q1,q3=np.nanquantile(log,[.25,.75],axis=0)
    scale=q3-q1
    scale=np.where(np.isfinite(scale)&(scale>0),scale,1.)
    buckets=defaultdict(list)
    rare=np.flatnonzero(y.to_numpy(float)>TAIL)
    labels=y.to_numpy(float)
    for i in rare:
        if np.isfinite(chem[i]).sum()<8:
            continue
        buckets[(str(groups.iloc[i]),str(sources.iloc[i]),tuple(coord[i]),tuple(cats[i]),
                 tuple(np.isnan(chem[i])))].append(int(i))
    pairs=set()
    for bucket in buckets.values():
        for i in bucket:
            near=[]
            for j in bucket:
                if j==i or abs(labels[i]-labels[j])>max(20.,.4*min(labels[i],labels[j])):
                    continue
                shared=np.isfinite(raw[i])&np.isfinite(raw[j])
                union=np.isfinite(raw[i])|np.isfinite(raw[j])
                if shared.sum()<8 or shared.sum()<.8*union.sum():
                    continue
                valid=np.isfinite(log[i])&np.isfinite(log[j])
                distance=float(np.sqrt(np.mean(((log[i,valid]-log[j,valid])/scale[valid])**2)))
                if 0<distance<=1.5:
                    near.append((distance,j))
            for _,j in sorted(near)[:5]:
                pairs.add(tuple(sorted((i,j))))
    return sorted(pairs)


def synthesize(x,y,groups,sources,canonical_columns,measured_columns,categorical,rng,gaussian=False):
    # x contains original canonical and admitted raw additional measurements;
    # never interpolate logarithmic ratios or engineered indices.
    canonical=[c for c in CHEM if c in canonical_columns]
    measured=canonical+list(measured_columns)
    if len(canonical)<8 or len(set(measured))!=len(measured):
        raise ValueError("Invalid chemistry schema")
    pairs=compatible_pairs(x,y,groups,sources,canonical,measured,categorical)
    budget=2*int(y.gt(TAIL).sum())
    raw=x[measured].to_numpy(float)
    oxide=np.array([c in OXIDES for c in measured])
    use,pair_use,group_use=Counter(),Counter(),Counter()
    rows,labels,trace=[],[],[]
    while len(rows)<budget:
        eligible=[(i,j) for i,j in pairs if use[i]<MAX_PARENT and use[j]<MAX_PARENT and pair_use[i,j]<MAX_PAIR]
        if not eligible:
            break
        least=min(group_use[str(groups.iloc[i])] for i,j in eligible)
        eligible=[(i,j) for i,j in eligible if group_use[str(groups.iloc[i])]==least]
        i,j=eligible[int(rng.integers(len(eligible)))]
        pair_use[i,j]+=1
        alpha=float(rng.uniform(.1,.9))
        a,b=raw[i],raw[j]
        common=np.isfinite(a)&np.isfinite(b)
        values=np.where(common,a+alpha*(b-a),np.nan)
        # SMOGN-inspired bounded perturbations of trace measurements only.
        # All values remain in their own parent envelopes; oxides and target
        # use the same convex coefficient. No claim of measured chemistry.
        if gaussian:
            noise=rng.normal(0.,.1,size=len(measured))*np.abs(a-b)
            values=np.where(common&~oxide,np.clip(values+noise,np.minimum(a,b),np.maximum(a,b)),values)
        if (np.isinf(values).any() or (values[np.isfinite(values)]<0).any()
                or (values[oxide&np.isfinite(values)]>100).any()):
            continue
        row=x.iloc[i].copy()
        row.loc[measured]=values
        # Preserve coordinates/categorical variables; all derived values replaced below.
        label=float(y.iloc[i]+alpha*(y.iloc[j]-y.iloc[i]))
        if not TAIL<label<=float(y.max()):
            raise ValueError("Synthetic target outside historical TRAIN tail")
        rows.append(row)
        labels.append(label)
        use[i]+=1
        use[j]+=1
        group_use[str(groups.iloc[i])]+=1
        trace.append({"parent_a":str(x.index[i]),"parent_b":str(x.index[j]),
            "lambda":alpha,"source":str(sources.iloc[i]),"group":str(groups.iloc[i]),
            "target":label,"action":"bounded_gaussian_interpolation" if gaussian else "interpolation"})
    if not rows:
        return x.iloc[:0],pd.Series(dtype=float),trace,{"compatible_pairs":len(pairs),"budget":budget}
    generated=pd.DataFrame(rows).reset_index(drop=True)
    canonical_frame=add_geochemical_features(generated[list(canonical_columns)],version=2)
    recover_cols=[c for c in measured_columns if c.startswith("georoc_")]
    extras=[c for c in measured_columns if c not in recover_cols]
    # Reuse the exact existing enrichment formula, rather than interpolating ratios.
    mm,rr=measurement_blocks(pd.concat([canonical_frame,generated[extras]],axis=1),
                              generated[recover_cols],canonical_frame)
    output=pd.concat([canonical_frame,mm,rr],axis=1).reindex(columns=x.columns)
    return output,pd.Series(labels),trace,{"compatible_pairs":len(pairs),"budget":budget}


def augment(x,y,groups,sources,canonical_columns,measured_columns,categorical,policy,seed):
    validate(x,y,groups,sources)
    if policy not in ARMS:
        raise ValueError("Unknown augmentation treatment")
    rng=np.random.default_rng(seed)
    trace=[]
    info={"budget":2*int(y.gt(TAIL).sum()),"compatible_pairs":None}
    if policy in ("rare_copies","relevance_copies"):
        pos=copies(y,groups,rng,policy)
        xx,yy=x.iloc[pos].copy(),y.iloc[pos].copy()
        trace=[{"parent_a":str(x.index[i]),"action":"copy","target":float(y.iloc[i]),
                "group":str(groups.iloc[i]),"source":str(sources.iloc[i])} for i in pos]
    elif policy in ("smoter_enriched","gaussian_enriched"):
        xx,yy,trace,info=synthesize(x,y,groups,sources,canonical_columns,measured_columns,
                                   categorical,rng,gaussian=policy=="gaussian_enriched")
    else:
        xx,yy=x.iloc[:0],y.iloc[:0]
    if len(xx)!=len(yy) or len(xx)!=len(trace):
        raise ValueError("Augmentation lineage incomplete")
    if len(xx):
        fit_x=pd.concat([x,xx],ignore_index=True)
        fit_y=pd.concat([y,yy],ignore_index=True)
    else:
        fit_x,fit_y=x.copy(),y.copy()
    w=weights(y,groups,sources,"density_05") if policy=="density_kde" else np.ones(len(fit_y))
    return fit_x,fit_y,w,trace,{**info,"real_rows":len(x),"fit_rows":len(fit_x),
        "added_rows":len(xx),"real_tail_rows":int(y.gt(TAIL).sum()),"fit_tail_rows":int(fit_y.gt(TAIL).sum())}
