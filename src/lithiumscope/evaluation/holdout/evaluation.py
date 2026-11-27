"""Reveal committed labels once, compare paired predictions, never promote."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.evaluation.holdout.registry import (
    normalized, read_sealed, read_csv, verified_registration, verify_artifact, write_sealed,
    UNKNOWN,
)


def _regression(y,p):
    from lithiumscope.core.scientific.regression_metrics import regression_metrics
    if np.unique(y).size<2:
        raise ValueError("Constant Li truth cannot support a comparative regression gate")
    return regression_metrics(y,p).to_dict()


def _paired_regression(data,iterations,seed):
    groups=normalized(data.spatial_group).to_numpy()
    blocks=[np.flatnonzero(groups==g) for g in np.unique(groups)]
    y=data.li_ppm.to_numpy(float)
    a=data.reference_prediction.to_numpy(float)
    b=data.candidate_prediction.to_numpy(float)
    rng=np.random.default_rng(seed)
    differences=[]
    for _ in range(iterations):
        take=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])
        ea=y[take]-a[take]
        eb=y[take]-b[take]
        differences.append([np.sqrt(np.mean(eb**2))-np.sqrt(np.mean(ea**2)),
                            np.mean(np.abs(eb))-np.mean(np.abs(ea))])
    intervals=np.quantile(differences,[.025,.975],axis=0)
    return {"groups":len(blocks),"iterations":iterations,"candidate_minus_reference_rmse_ci95":intervals[:,0].tolist(),
            "candidate_minus_reference_mae_ci95":intervals[:,1].tolist(),
            "scope":"paired cluster bootstrap, conditional on the sampled sites"}


def evaluate_registered(root,labels_path):
    root=Path(root).resolve()
    if any((root/name).exists() for name in ("label_reveal.json","evaluation.json","paired_evaluation.csv")):
        raise FileExistsError("Label reveal/evaluation already recorded; do not optimize on this holdout")
    registration,sites=verified_registration(root)
    frozen=read_sealed(root/"scores.json")
    if frozen["registration_sha256"]!=file_sha256(root/"registration.json"):
        raise ValueError("Predictions belong to another registration")
    paths={arm:verify_artifact(record) for arm,record in frozen["predictions"].items()}
    labels_path=Path(labels_path)
    if file_sha256(labels_path)!=registration["label_commitment_sha256"]:
        raise ValueError("Revealed labels differ from the preregistered commitment")
    # Consume this registered evaluation even if revealed labels prove invalid.
    write_sealed(root/"label_reveal.json",{
        "revealed_at_utc":datetime.now(timezone.utc).isoformat(),
        "label_sha256":registration["label_commitment_sha256"],
        "scores_sha256":file_sha256(root/"scores.json")})
    labels=read_csv(labels_path)
    required={"sample_id","li_ppm","unit","analytical_method","source_id"}
    if not required<=set(labels) or labels.sample_id.duplicated().any():
        raise ValueError("Labels must identify each registered sample and method")
    if set(labels.sample_id)!=set(sites.sample_id):
        raise ValueError("Label identity/coverage differs from frozen cohort")
    if not normalized(labels.unit).eq("ppm").all() or normalized(labels.analytical_method).isin(UNKNOWN).any():
        raise ValueError("Label unit or analytical method is unverified")
    labels["li_ppm"]=pd.to_numeric(labels.li_ppm,errors="coerce")
    if not np.isfinite(labels.li_ppm).all() or (labels.li_ppm<0).any():
        raise ValueError("Labels must be finite nonnegative ppm")
    data=sites[["sample_id","spatial_group","region","source_id"]].merge(
        labels[list(required)].rename(columns={"source_id":"label_source_id"}),on="sample_id",validate="one_to_one")
    if not normalized(data.source_id).eq(normalized(data.label_source_id)).all():
        raise ValueError("Label sources differ from registered sample provenance")
    for arm,path in paths.items():
        prediction=read_csv(path)
        if set(prediction.sample_id)!=set(sites.sample_id) or prediction.sample_id.duplicated().any():
            raise ValueError("Paired prediction coverage does not match frozen sites")
        data=data.merge(prediction.rename(columns={c:f"{arm}_{c}" for c in prediction if c!="sample_id"}),
                        on="sample_id",validate="one_to_one")
        data[f"{arm}_prediction"]=pd.to_numeric(data[f"{arm}_prediction"],errors="coerce")
    y=data.li_ppm.to_numpy(float)
    a=data.reference_prediction.to_numpy(float)
    b=data.candidate_prediction.to_numpy(float)
    valid_a=np.isfinite(a)&(a>=0)
    valid_b=np.isfinite(b)&(b>=0)
    if registration["model_group"]=="model_2":
        valid_a &= a<=1
        valid_b &= b<=1
    evidence={"evaluated_at_utc":datetime.now(timezone.utc).isoformat(),"registration_sha256":file_sha256(root/"registration.json"),
              "scores_sha256":file_sha256(root/"scores.json"),"label_sha256":file_sha256(labels_path),
              "n":len(data),"reference_valid":int(valid_a.sum()),"candidate_valid":int(valid_b.sum()),
              "paired_valid":int((valid_a&valid_b).sum()),"scientific_release_evidence_pass":False,
              "automatic_model_activation":False,"requires_independent_human_review":True,
              "scope":"Prospective local evaluation, conditional on supplied documentary attestations. No activation; no reuse for selection."}
    if not (valid_a&valid_b).all():
        evidence.update(state="blocked_incomplete_predictions",reason="No subset-only comparison; abstentions retained")
    elif registration["model_group"]=="model_1":
        from lithiumscope.evaluation.candidates.gates import model_1_gate
        if np.unique(y).size<2:
            evidence.update(state="blocked_constant_truth")
        else:
            m1,m2=_regression(y,a),_regression(y,b)
            passed,reasons=model_1_gate(m1,m2)
            evidence.update(state="ready_for_review",reference_metrics=m1,candidate_metrics=m2,
                            provisional_numeric_gate=passed,numeric_gate_reasons=reasons,
                            paired_uncertainty=_paired_regression(data,**registration["bootstrap"]))
    else:
        from lithiumscope.core.scientific.classification_metrics import classification_metrics
        from lithiumscope.model_2.diagnostics.paired_uncertainty import bootstrap_paired_groups
        from lithiumscope.model_2.evaluation.release_gate import assess_release, ExternalGatePolicy
        label=(y>=registration["label_cutoff_ppm"]).astype(int)
        if np.unique(label).size!=2:
            evidence.update(state="blocked_single_class")
        else:
            metrics={}
            frames={}
            for arm,p in (("reference",a),("candidate",b)):
                cuts=pd.to_numeric(data[f"{arm}_operating_threshold"],errors="coerce")
                if cuts.nunique()!=1 or not np.isfinite(cuts).all():
                    raise ValueError("Operating threshold changed within the frozen evaluation")
                metrics[arm]={"conventional":classification_metrics(label,p),
                              "operating":classification_metrics(label,p,float(cuts.iloc[0]))}
                frames[arm]=pd.DataFrame({"sample_index":data.sample_id,"spatial_group":data.spatial_group,
                                          "y_true":label,"score":p})
            uncertainty=bootstrap_paired_groups(frames["reference"],frames["candidate"],**registration["bootstrap"])
            ref,can=metrics["reference"]["conventional"],metrics["candidate"]["conventional"]
            # Attestation fields are not machine-certified: numerical review only.
            gate=assess_release({"n":len(data),"positives":int(label.sum()),"negatives":int((1-label).sum()),
                "regions":int(normalized(data.region).nunique()),"possible_duplicate_count":0,
                "candidate_auc":can["roc_auc"],"reference_auc":ref["roc_auc"],
                "candidate_ap":can["average_precision"],"reference_ap":ref["average_precision"],
                "candidate_brier":can["brier_score"],"reference_brier":ref["brier_score"],
                "paired_auc_gain_ci_lower":uncertainty["metrics"]["roc_auc"]["ci95"][0],
                "external_independence_verified":False,"holdout_unconsulted_during_model_selection":False,
                "sample_identity_and_provenance_verified":False,"labels_field_verified":False},
                policy=ExternalGatePolicy(**registration["m2_gate_policy"]))
            evidence.update(state="ready_for_review",metrics=metrics,paired_uncertainty=uncertainty,
                            release_gate=gate)
    with (root/"paired_evaluation.csv").open("x",encoding="utf-8",newline="") as handle:
        data.to_csv(handle,index=False)
    evidence["paired_table_sha256"]=file_sha256(root/"paired_evaluation.csv")
    evidence["label_reveal_sha256"]=file_sha256(root/"label_reveal.json")
    write_sealed(root/"evaluation.json",evidence)
    return evidence
