"""Version-aware raw-patch scenarios with verified baseline and honest abstentions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.experiments.scenarios.imagery import image_catalog, perturb_image
from lithiumscope.model_2.prediction.spectral_contract import prepare_spectral_input, SpectralAbstention
from lithiumscope.model_2.preparation.imagery import load_imagery
from lithiumscope.model_2.preparation.image_preprocessing import preprocess_image
from lithiumscope.model_2.preparation.features.spectral import extract_spectral_features, DEFAULT_BAND_NAMES
from lithiumscope.model_2.schema import out_of_range_fraction


def evaluate_verified_image(bundle,path,destination,seed=42):
    scenarios=image_catalog(seed)
    rows=[]
    try:
        baseline,_=prepare_spectral_input(path,bundle,allow_acquisition=False)
        if bundle.get("patch_input_contract") is not None:
            from lithiumscope.model_2.preparation.patch_contract import load_verified_patch
            raw,_,_=load_verified_patch(path,bundle["patch_input_contract"])
        else:
            raw=load_imagery(path)
    except SpectralAbstention as exc:
        rows=[{"scenario":s.name,"kind":s.kind,"intensity":s.intensity,"status":exc.status,
               "reason":str(exc),"score":np.nan,"ood_fraction":np.nan} for s in scenarios]
    else:
        expected=list(bundle["features"])
        bands=tuple(bundle.get("band_names",bundle.get("base_band_names",DEFAULT_BAND_NAMES)))
        v4=int(bundle.get("feature_extractor_version",3))==4
        cutoff=bundle.get("quality_min_joint_valid")
        for scenario in scenarios:
            disturbed=perturb_image(raw,scenario)
            coverage=float(np.isfinite(disturbed).all(axis=0).mean())
            row={"scenario":scenario.name,"kind":scenario.kind,"intensity":scenario.intensity,
                 "joint_valid_fraction":coverage,"scope":"base_6_band_perturbation_fixed_verified_rededge" if v4 else "base_band_perturbation",
                 "score":np.nan,"ood_fraction":np.nan}
            if not np.isfinite(disturbed).any(axis=(1,2)).all() or (cutoff is not None and coverage<float(cutoff)):
                row["status"]="ABSTAIN_LOW_SPECTRAL_COVERAGE"
            else:
                clean=preprocess_image(disturbed,normalize_per_band=bool(bundle.get("normalize_per_band",False)))
                changed=extract_spectral_features(clean,band_names=bands)
                frame=baseline.copy()
                for name,value in changed.items():
                    if name in frame:
                        frame[name]=value
                if scenario.kind=="baseline":
                    frame=baseline
                if not np.isfinite(frame[expected].to_numpy(float)).all():
                    row["status"]="ABSTAIN_NONFINITE_FEATURES"
                else:
                    score=float(bundle["estimator"].predict_proba(frame[expected])[0,1])
                    if not np.isfinite(score) or not 0<=score<=1:
                        raise ValueError("Invalid score")
                    row.update(status="SCORED",score=score,ood_fraction=out_of_range_fraction(frame[expected],bundle.get("applicability_profile",{})))
            rows.append(row)
    table=pd.DataFrame(rows)
    table["absolute_score_shift"]=(table.score-table.loc[table.scenario=="baseline","score"].iloc[0]).abs()
    destination.mkdir(parents=True,exist_ok=True)
    output=destination/"image_scenario_metrics.csv"
    table.to_csv(output,index=False)
    return output
