"""Fixed local analog estimator; no target-dependent distance or TEST fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lithiumscope.experiments.data_lab.candidate_screening import CHEM

K=5
MIN_SHARED=8
GEOGRAPHIC_SCALE_KM=50.
MIN_DISTANCE=.1


def _matrix(frame, columns):
    a=frame[columns].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    return np.log1p(np.where(np.isfinite(a)&(a>=0),a,np.nan))


def _coordinates(frame):
    coords=frame[["Latitude","Longitude"]].to_numpy(float)
    if not np.isfinite(coords).all() or (np.abs(coords[:,0])>90).any() or (np.abs(coords[:,1])>180).any():
        raise ValueError("Invalid coordinates")
    return np.deg2rad(coords)


def fit_analog(x,y):
    if not x.index.is_unique or not x.index.equals(y.index):
        raise ValueError("Invalid analog TRAIN identities")
    labels=y.to_numpy(float)
    if not np.isfinite(labels).all() or (labels<0).any() or not len(labels):
        raise ValueError("Invalid analog TRAIN labels")
    candidates=[c for c in CHEM if c in x]
    matrix=_matrix(x,candidates)
    median=np.nanmedian(matrix,axis=0)
    q25,q75=np.nanquantile(matrix,[.25,.75],axis=0)
    spread=q75-q25
    active=np.isfinite(median)&np.isfinite(spread)&(spread>0)&(np.isfinite(matrix).mean(axis=0)>=.5)
    cols=[c for c,v in zip(candidates,active,strict=True) if v]
    if len(cols)<MIN_SHARED:
        raise ValueError("Too few independently measured chemical variables")
    median,spread=median[active],spread[active]
    train=(matrix[:,active]-median)/spread
    coord=_coordinates(x)
    train_ids=x.index.astype(str).to_numpy()
    if len(set(train_ids))!=len(train_ids):
        raise ValueError("Ambiguous normalized analog TRAIN IDs")

    def predict(frame,fallback):
        if not frame.index.is_unique or set(frame.index.astype(str))&set(train_ids):
            raise ValueError("Analog evaluation overlaps TRAIN")
        fallback=np.asarray(fallback,float)
        if fallback.shape!=(len(frame),) or not np.isfinite(fallback).all() or (fallback<0).any():
            raise ValueError("Invalid analog fallback")
        query=(_matrix(frame,cols)-median)/spread
        qcoord=_coordinates(frame)
        result=fallback.copy()
        trace=[]
        for j,(row,loc) in enumerate(zip(query,qcoord,strict=True)):
            shared=np.isfinite(train)&np.isfinite(row)
            count=shared.sum(axis=1)
            delta=np.where(shared,train-row,0.)
            squared=np.divide((delta**2).sum(axis=1),count,
                out=np.full(len(train),np.inf),where=count>=MIN_SHARED)
            dlat=coord[:,0]-loc[0]
            dlon=coord[:,1]-loc[1]
            hav=np.sin(dlat/2)**2+np.cos(loc[0])*np.cos(coord[:,0])*np.sin(dlon/2)**2
            km=6371.*2*np.arcsin(np.sqrt(np.clip(hav,0,1)))
            # Reject near-identical chemistry at the same site without inspecting Li.
            duplicate=(km<=.1)&(squared<=1e-4)
            distance=np.sqrt(squared+np.log1p(km/GEOGRAPHIC_SCALE_KM)**2
                             +.25*(1-count/len(cols)))
            distance[(count<MIN_SHARED)|duplicate]=np.inf
            order=np.argsort(distance,kind="stable")[:K]
            order=order[np.isfinite(distance[order])]
            if len(order)>=3:
                weights=1./np.maximum(distance[order],MIN_DISTANCE)**2
                result[j]=float(np.average(labels[order],weights=weights))
                neighbors=[{"row_id":str(train_ids[i]),"distance":float(distance[i]),"km":float(km[i]),
                            "shared_chemistry":int(count[i])} for i in order]
            else:
                neighbors=[]
            trace.append({"row_id":str(frame.index[j]),"fallback":not bool(neighbors),"neighbors":neighbors,
                          "near_duplicate_exclusions":int(duplicate.sum())})
        return result,trace

    return predict,{"columns":cols,"k":K,"min_shared":MIN_SHARED,"geographic_scale_km":GEOGRAPHIC_SCALE_KM}
