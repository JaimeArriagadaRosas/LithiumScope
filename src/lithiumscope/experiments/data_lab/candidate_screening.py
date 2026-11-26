"""Conservative duplicate and metadata audit across candidate and frozen base data.

Flags are evidence for review, not proof of duplication or training eligibility.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CHEM = ("SiO2","TiO2","Al2O3","Fe2O3","MnO","MgO","CaO","Na2O","K2O","P2O5",
        "Th_icpms","U_icpms","Rb_icpms","Cs_icpms","Nb_icpms","Ta_icpms","Pb_icpms",
        "Ba_icpms","Sr_icpms","Zr_icpms","V_icpms","Hf_icpms")


def text(frame, name):
    return (frame.get(name,pd.Series("",index=frame.index)).fillna("").astype(str)
            .str.strip().str.casefold().replace({"nan":"","none":"","unknown":"","<na>":""}))


def sample(frame):
    return (text(frame,"source_sample").where(text(frame,"source_sample").ne(""),text(frame,"Sample"))
            .str.replace(r"\s*\[\d+\]\s*$","",regex=True).str.replace(r"[^a-z0-9]","",regex=True))


def num(frame,names):
    return frame.reindex(columns=list(names)).apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)


def coords(frame):
    return num(frame,["Latitude","Longitude"])


def group(frame):
    xy=coords(frame)
    valid=xy.Latitude.between(-90,90)&xy.Longitude.between(-180,180)
    result=pd.Series("",index=frame.index)
    result.loc[valid]="g_"+np.floor((xy.loc[valid,"Latitude"]+90)/.5).astype(int).astype(str)+"_"+np.floor((xy.loc[valid,"Longitude"]+180)/.5).astype(int).astype(str)
    return result


def _duplicate_flags(x,base):
    n=len(x)
    all_rows=pd.concat([x,base],ignore_index=True,sort=False)
    xy=coords(all_rows).to_numpy(float)
    li=num(all_rows,["Li_icpms"]).Li_icpms.to_numpy(float)
    chem=num(all_rows,CHEM).to_numpy(float)
    valid=np.isfinite(xy).all(axis=1)&(np.abs(xy[:,0])<=90)&(np.abs(xy[:,1])<=180)&np.isfinite(li)&(li>=0)
    sample_names=sample(all_rows).to_numpy(str)
    ids=text(all_rows,"source_unique_id")
    pub=text(all_rows,"source_publication")
    pub_sample=pub+"|"+pd.Series(sample_names)
    exact_id=ids.ne("")&ids.duplicated(keep=False)
    exact_pub=pub.ne("")&pd.Series(sample_names).ne("")&pub_sample.duplicated(keep=False)
    coordinate_key=pd.DataFrame({"lat":np.round(xy[:,0],5),"lon":np.round(xy[:,1],5),"li":np.round(li,6)})
    exact_site=valid&coordinate_key.duplicated(keep=False).to_numpy()
    near=np.zeros(n,dtype=bool)
    ix=np.flatnonzero(valid)
    tree=cKDTree(xy[ix])
    for i in np.flatnonzero(valid[:n]):
        for j in ix[tree.query_ball_point(xy[i],.001,p=np.inf)]:
            if i==j or abs(li[i]-li[j])>max(.05,.01*max(li[i],li[j])):
                continue
            shared=np.isfinite(chem[i])&np.isfinite(chem[j])
            same_sample=bool(sample_names[i]) and sample_names[i]==sample_names[j]
            similar=False
            if shared.sum()>=5:
                a,b=chem[i,shared],chem[j,shared]
                similar=bool(np.all(np.abs(a-b)<=np.maximum(.001,.02*np.maximum(np.abs(a),np.abs(b)))))
            if same_sample or similar:
                near[i]=True
                break
    return {"target_and_coords_ok":valid[:n],"duplicate_id":exact_id.iloc[:n].to_numpy(),
            "duplicate_publication_sample":exact_pub.iloc[:n].to_numpy(),
            "duplicate_coordinate_li":exact_site[:n],"possible_near_duplicate":near}


def screen(candidates,base,known_groups):
    x=candidates.reset_index(drop=True).copy()
    base=base.reset_index(drop=True)
    out=x[["source_dataset","source_file","source_row_position","source_unique_id",
           "source_publication","source_sample","Rock_type","source_tectonic_setting",
           "Latitude","Longitude","Li_icpms"]].copy()
    for key,value in _duplicate_flags(x,base).items():
        out[key]=value
    out["spatial_group"]=group(x)
    out["existing_spatial_group"]=out.spatial_group.isin(set(known_groups))
    out["quality_ok"]=num(x,["SUM (no water)"]).iloc[:,0].between(94,102)
    out["material_ok"]=text(x,"Sample_type").eq("whole rock")
    raw=num(x,["source_li_raw"]).iloc[:,0]
    y=num(x,["Li_icpms"]).iloc[:,0]
    out["lithium_unit_and_raw_ok"]=text(x,"source_li_unit").eq("ppm")&np.isclose(raw,y,rtol=1e-8,atol=1e-8,equal_nan=False)
    out["geochemical_predictors_present"]=num(x,CHEM).notna().sum(axis=1)
    out["publication_and_sample_present"]=text(x,"source_publication").ne("")&sample(x).ne("")
    out["method_known"]=text(x,"source_li_method").ne("")
    out["crs_known"]=text(x,"source_coordinate_crs").ne("")
    eligible=(out.quality_ok&out.target_and_coords_ok&out.material_ok&out.lithium_unit_and_raw_ok
              &out.publication_and_sample_present&(out.geochemical_predictors_present>=8)
              &~out[["duplicate_id","duplicate_publication_sample","duplicate_coordinate_li",
                     "possible_near_duplicate"]].any(axis=1))
    convergent=text(x,"source_tectonic_setting").eq("convergent margin")
    high=y.gt(696)
    out["screening_state"]=np.select([~eligible,eligible&convergent,eligible&high],
        ["excluded_screening","convergent_review","high_transfer_review"],default="other_domain_review")
    out["metadata_review_required"]=~(out.method_known&out.crs_known)
    out["approved_for_training"]=False
    return out
