"""EDA preparation: dry-run actual TRAIN treatments, never fit predictors."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.experiments.generalization.arms import (
    M1_ARMS, M2_ARMS, prepare_regression_arm, prepare_classification_arm,
)
from lithiumscope.experiments.generalization.partitions import partitions, training_target
from lithiumscope.experiments.generalization.prepared import load_snapshot


def _json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _identity(directory, cfg):
    from importlib.metadata import version
    names = ("preflight.py", "partitions.py", "arms.py", "resampling.py", "signature_policy.py", "domain.py")
    return canonical_json_hash({"snapshot": file_sha256(directory/"manifest.json"), "config": cfg,
        "code": {n: file_sha256(Path(__file__).with_name(n)) for n in names},
        "versions": {p: version(p) for p in ("numpy","pandas","scikit-learn")}})


def _validate(directory, key):
    record = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    expected = {"partitions.json", "treatments.csv", "augmentation.csv", "fold_summary.csv"}
    if (directory.is_symlink() or (directory/"manifest.json").is_symlink()
            or record.get("identity") != key or record.get("state") not in {"complete","partial","blocked"}
            or set(record.get("artifacts", {})) != expected):
        raise ValueError("Invalid treatment preparation manifest")
    for name, digest in record["artifacts"].items():
        path = directory/name
        if path.is_symlink() or file_sha256(path) != digest:
            raise ValueError("Treatment preparation integrity failure")
    return record


def _prepare_one(group, arm, data, cfg, train, test, seed):
    x,y,groups,sources,eligible,numeric,categorical = data
    fit, labels, threshold = training_target(group,data,cfg,train)
    xx, gg = x.iloc[fit], groups.iloc[fit]
    if group == "model_1":
        matrix, target, _, schema, trace = prepare_regression_arm(arm,seed,xx,labels,gg,sources.iloc[fit],numeric,categorical)
        features = schema.numeric + schema.categorical
        # No imputer fitted here: canonical M1 estimator owns it, including
        # TabNet's own spatial early-stopping split.
        if arm == "knn_imputation":
            from lithiumscope.model_1.training.preprocessing.tabular import build_preprocessor
            transformed = build_preprocessor(schema,"random_forest").fit_transform(matrix,target)
            if not np.isfinite(transformed).all():
                raise ValueError("KNN imputation produced nonfinite TRAIN")
            imputed_digest = canonical_json_hash({"matrix":transformed.tolist()})
        else:
            imputed_digest = None
        fit_digest = canonical_json_hash({"imputed":imputed_digest,"rows": pd.util.hash_pandas_object(matrix,index=False).astype(str).tolist(),
                                           "target": np.asarray(target,float).tolist(), "features": features})
    else:
        matrix,target,_,_,features,trace,weights = prepare_classification_arm(arm,seed,xx,labels,gg,return_weights=True)
        if not np.isfinite(matrix).all() or set(np.unique(target)) != {0,1}:
            raise ValueError("Invalid prepared classification matrix")
        fit_digest = canonical_json_hash({"rows": np.asarray(matrix).tolist(),"target": np.asarray(target).tolist(),"features": features,"weights":None if weights is None else weights.tolist()})
    fit_ids = set(xx.index.astype(str))
    for entry in trace:
        for name in ("row","parent_a","parent_b"):
            if name in entry and entry[name] not in fit_ids:
                raise ValueError("Treatment provenance escapes eligible TRAIN")
    result = {"fit_before":len(fit),"fit_after":len(target),"test_reserved":len(test),
        "features_before":len(x.columns),"features_after":len(features),
        "feature_names":json.dumps(features,ensure_ascii=False),
        "fit_sha256":fit_digest,"threshold_ppm":threshold,
        "generated_rows":max(0,len(target)-len(fit)),"removed_rows":max(0,len(fit)-len(target))}
    if group == "model_1":
        result.update(target_min=float(np.min(target)),target_max=float(np.max(target)))
    else:
        result.update(positive_before=int(labels.sum()),positive_after=int(np.sum(target)),
            effective_fit_weight=float(len(target) if weights is None else weights.sum()),
            effective_positive_weight=float(np.sum(target) if weights is None else np.dot(target,weights)))
    return result,trace


def prepare_treatments(group, snapshot, cfg):
    snapshot=Path(snapshot)
    data=load_snapshot(snapshot,snapshot.name)
    key=_identity(snapshot,cfg)
    destination=snapshot/"treatment_plans"/key
    if destination.exists():
        return destination,_validate(destination,key)
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=destination.parent/(".staging-"+uuid4().hex)
    staging.mkdir()
    try:
        plan=partitions(group,data,cfg)
        x,y,groups,_,eligible,_,_=data
        arms=M1_ARMS if group=="model_1" else M2_ARMS
        rows,traces,fold_rows,split_rows=[],[],[],[]
        for outer,(tr,te,inner) in enumerate(plan,1):
            split_rows.append({"fold":outer,"train":tr.tolist(),"test":te.tolist(),
                "inner":[{"train":a.tolist(),"validation":b.tolist()} for a,b in inner]})
            for inner_number,(train,test) in enumerate([(tr,te),*inner]):
                seed=42+outer if inner_number==0 else 42+outer*10+inner_number
                from lithiumscope.experiments.generalization.domain import support_summary
                fit_indices = train[eligible[train]]
                diagnostic = support_summary(x.iloc[fit_indices],x.iloc[test],data[5],data[6])
                diagnostic["reserved_target_max"] = float(y.iloc[test].max())
                diagnostic["reserved_above_train_target_max"] = int((y.iloc[test] > y.iloc[fit_indices].max()).sum())
                if group == "model_2":
                    from lithiumscope.experiments.generalization.signature_policy import signature_summary, signature_overlap
                    _, labels, threshold = training_target(group,data,cfg,train)
                    diagnostic.update({"train_"+k:v for k,v in signature_summary(x.iloc[fit_indices],labels).items()})
                    diagnostic.update({"reserved_"+k:v for k,v in signature_summary(x.iloc[test],(y.iloc[test]>=threshold).astype(int)).items()})
                    diagnostic["exact_signature_overlap_rows"] = signature_overlap(x.iloc[fit_indices],x.iloc[test])
                fold_rows.append({**diagnostic,"outer_fold":outer,"inner_fold":inner_number,
                    "train_rows":len(train),"qa_eligible_train":int(eligible[train].sum()),
                    "reserved_rows":len(test),"train_groups":int(groups.iloc[train].nunique()),
                    "reserved_groups":int(groups.iloc[test].nunique()),
                    "train_target_max":float(y.iloc[train[eligible[train]]].max())})
                for arm in arms:
                    context={"outer_fold":outer,"inner_fold":inner_number,"arm":arm,"seed":seed}
                    try:
                        result,trace=_prepare_one(group,arm,data,cfg,train,test,seed)
                        rows.append({**context,"state":"ready",**result})
                        traces.extend({**context,**entry} for entry in trace)
                    except ValueError as exc:
                        rows.append({**context,"state":"blocked","reason":str(exc)})
                print(f"  Preparación {group} | fold {outer}/{len(plan)} | "
                      f"{'TRAIN exterior' if inner_number==0 else 'TRAIN interno '+str(inner_number)}",flush=True)
        table=pd.DataFrame(rows)
        baseline=table.loc[table.arm.eq("reference")]
        state=("blocked" if not baseline.state.eq("ready").all() else
               "complete" if table.state.eq("ready").all() else "partial")
        _json(staging/"partitions.json",split_rows)
        table.to_csv(staging/"treatments.csv",index=False)
        pd.DataFrame(traces,columns=["outer_fold","inner_fold","arm","seed","row","action","parent_a","parent_b","lambda","group","source","weight","signature"]).to_csv(staging/"augmentation.csv",index=False)
        pd.DataFrame(fold_rows).to_csv(staging/"fold_summary.csv",index=False)
        summary=table.groupby(["arm","state"]).size().unstack(fill_value=0).to_dict(orient="index")
        manifest={"identity":key,"state":state,"model_group":group,"prepared_identity":snapshot.name,
            "model_fits":0,"scope":"train_only_treatment_preflight",
            "partition_count":len(fold_rows),"checks":len(table),"ready":int(table.state.eq("ready").sum()),
            "blocked":int(table.state.eq("blocked").sum()),"summary":summary,
            "artifacts":{p.name:file_sha256(p) for p in staging.iterdir()}}
        _json(staging/"manifest.json",manifest)
        staging.rename(destination)
        return destination,manifest
    except BaseException:
        import shutil
        shutil.rmtree(staging,ignore_errors=True)
        raise


def show_treatments(path, manifest):
    table=pd.read_csv(path/"treatments.csv")
    print("\n--- PREPARACIÓN DE TRATAMIENTOS SOLO TRAIN (sin entrenar predictores) ---")
    print(f"Particiones: {manifest['partition_count']} | verificaciones: {manifest['checks']} | "
          f"listas: {manifest['ready']} | bloqueadas: {manifest['blocked']}")
    for arm, subset in table.groupby("arm",sort=False):
        good=subset.loc[subset.state.eq("ready")]
        detail=""
        if not good.empty:
            detail=f" | filas de ajuste: {int(good.fit_after.min())}-{int(good.fit_after.max())} | variables: {int(good.features_after.min())}-{int(good.features_after.max())}"
        print(f"  {arm:24} listas {len(good):2}/{len(subset):2}{detail}")
        if len(good)!=len(subset):
            for reason,count in subset.loc[subset.state.ne("ready"),"reason"].value_counts().items():
                print(f"    BLOQUEADO ({count}): {reason}")
    folds = pd.read_csv(path/"fold_summary.csv")
    outer = folds.loc[folds.inner_fold.eq(0)]
    print("\nDiagnóstico descriptivo exterior (no utilizado para seleccionar tratamientos):")
    columns = ["outer_fold","train_target_max","reserved_target_max","reserved_above_train_target_max",
               "outside_numeric_range_rows","unseen_categories_rows","exact_signature_overlap_rows",
               "train_conflicting_rows","reserved_conflicting_rows","reserved_unavoidable_errors_same_signature"]
    print(outer[[c for c in columns if c in outer]].to_string(index=False))
    print("Ninguna fila sintética se agrega al dataset original ni a TEST.")
    print("La eficacia predictiva requiere evaluación posterior; las matrices aprendidas se ajustan por TRAIN.")
    print(f"Manifiesto de tratamientos: {path/'manifest.json'}")
