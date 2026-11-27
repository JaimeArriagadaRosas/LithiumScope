"""Compare a reconstructed historical protocol, never mislabel it as the fitted release."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.experiments.integrated.compare_vnext_trained_runs import (
    classification_metrics, group_bootstrap_gain,
)
from lithiumscope.model_2.training.postprocess import (
    OLD_OOF, EXPECTED_OLD_REFERENCE_SHA256, PREVIOUS_ARM,
)
from lithiumscope.results.atomic_exports import write_table, write_json

CUTOFF_PPM=21.6
REFERENCE_RUN="training_20260920_195933_m0300"
REFERENCE_DATASET="b04b1291a8a7cbf24538253e521a93f32ff8c765ff437bd7c2ba5666a095121e"


def pair_reference(old,new):
    old=old.loc[old.arm.eq(PREVIOUS_ARM)].sort_values("test_index").reset_index(drop=True)
    new=new.sort_values("row_position").reset_index(drop=True)
    if len(old)!=706 or len(new)!=706:
        raise ValueError("Reconstructed protocol requires 706 historical cases")
    if (old.test_index.tolist()!=list(range(706)) or new.row_position.tolist()!=list(range(706))
            or old.sample_id.isna().any() or new.sample_id.isna().any()
            or old.sample_id.duplicated().any() or new.sample_id.duplicated().any()):
        raise ValueError("Missing or duplicated M2 case identity")
    for a,b in (("sample_id","sample_id"),("spatial_group","spatial_group"),("fold","outer_fold")):
        if not old[a].equals(new[b]):
            raise ValueError("M2 reconstructed reference differs in "+a)
    if not np.allclose(old.li_ppm,new.li_ppm,atol=1e-8,rtol=0,equal_nan=False):
        raise ValueError("M2 reconstructed reference differs in measured lithium")
    if old.groupby("spatial_group").fold.nunique().max()!=1:
        raise ValueError("M2 spatial group spans multiple outer folds")
    site=new.latitude.map(lambda v:f"{float(v):.5f}")+"|"+new.longitude.map(lambda v:f"{float(v):.5f}")
    if not old.site_screening_id.equals(site):
        raise ValueError("M2 reconstructed reference differs in coordinates")
    return pd.DataFrame({"sample_id":old.sample_id,"row_position":old.test_index,
                         "group":old.spatial_group,"fold":old.fold,"li_ppm":old.li_ppm,
                         "truth":old.li_ppm.ge(CUTOFF_PPM).astype(int),
                         "reference":old.score,"candidate":new.prospectivity_score})


def compare_reconstructed(old_identity,new_identity,old_run,new_run,destination,*,root):
    if old_identity["run_id"]!=REFERENCE_RUN or old_identity["dataset_sha256"]!=REFERENCE_DATASET:
        raise ValueError("Reconstructed protocol is specific to the September release")
    reference=root/OLD_OOF
    if file_sha256(reference)!=EXPECTED_OLD_REFERENCE_SHA256:
        raise ValueError("Reconstructed reference SHA-256 mismatch")
    old=pd.read_csv(reference)
    candidate_path=new_run/"tables"/("oof_predictions_"+new_identity["algorithm"]+".csv")
    new=pd.read_csv(candidate_path)
    paired=pair_reference(old,new)
    published_path=old_run/"tables"/("oof_predictions_"+old_identity["algorithm"]+".csv")
    published=pd.read_csv(published_path)
    published_metrics=classification_metrics(published.y_true,published.prospectivity_score)
    candidate_native=classification_metrics(new.y_true,new.prospectivity_score)
    old_metrics=classification_metrics(paired.truth,paired.reference)
    new_metrics=classification_metrics(paired.truth,paired.candidate)
    gains=[group_bootstrap_gain(paired.truth,paired.reference,paired.candidate,paired.group,
                               metric=m,repeats=5000) for m in ("roc_auc","average_precision","brier")]
    destination.mkdir(parents=True,exist_ok=True)
    write_table(destination/"paired_oof.csv",paired)
    return write_json(destination/"comparison.json",{
        "scope":"reconstructed_release_protocol_not_published_artifact",
        "release":old_identity,"candidate":new_identity,
        "comparison":{"matched":len(paired),"same_outer_fold":len(paired),
                      "protocol":"reconstructed_reference_fixed_li_cutoff",
                      "fixed_li_cutoff_ppm":CUTOFF_PPM,
                      "published_artifact_paired":False},
        "reference_sha256":file_sha256(reference),
        "candidate_oof_sha256":file_sha256(candidate_path),
        "published_oof_sha256":file_sha256(published_path),
        "published_descriptive_metrics":published_metrics,
        "candidate_native_metrics":candidate_native,
        "reference_metrics":old_metrics,"candidate_metrics":new_metrics,"gains":gains,
        "independent_holdout":False})
