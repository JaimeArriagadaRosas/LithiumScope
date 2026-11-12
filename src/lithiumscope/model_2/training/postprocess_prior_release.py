"""Separated scientific M2 report; imported lazily for API compatibility."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_old_release_comparison(root,manifest,folds,scores,output):
    from lithiumscope.model_2.training.postprocess import (
        OLD_OOF,
        OLD_META,
        PREVIOUS_ARM,
        EXPECTED_OLD_REFERENCE_SHA256,
        EXPECTED_OLD_METADATA_SHA256,
        get_json,
        save_json,
        metrics,
        bootstrap_group_delta,
        file_sha256,
        ranking_evaluation,
    )
    if file_sha256(root/OLD_OOF)!=EXPECTED_OLD_REFERENCE_SHA256:
        raise ValueError("Saved prior-release-protocol prediction source has changed")
    if file_sha256(root/OLD_META)!=EXPECTED_OLD_METADATA_SHA256:
        raise ValueError("Published release metadata has changed")
    meta=get_json(root/OLD_META)
    if meta.get("algorithm")!="catboost" or meta.get("samples")!=706:
        raise ValueError("September release reference has changed")
    cutoff=float(meta["threshold_ppm"])
    old=pd.read_csv(root/OLD_OOF)
    old=old[old.arm==PREVIOUS_ARM].sort_values("test_index")
    if len(old)!=706 or old.test_index.tolist()!=list(range(706)):
        raise ValueError("Old retrained release protocol lacks 706 heldout predictions")
    if not np.array_equal(
        old.sample_id.astype(str),manifest.sample_id.iloc[:706].astype(str)):
        raise ValueError("Old release protocol sample IDs differ from new competition")
    if not np.allclose(old.li_ppm,manifest.Li_icpms.iloc[:706]):
        raise ValueError("Original 706 Li values differ from reference")
    if not np.array_equal(old.fold.to_numpy(int),folds[:706]):
        raise ValueError("Original 706 spatial heldout folds no longer match")
    label=(manifest.Li_icpms.iloc[:706].to_numpy(float)>=cutoff).astype(int)
    common={
        "test_index":np.arange(706),"fold":folds[:706],
        "sample_id":manifest.sample_id.iloc[:706].astype(str).to_numpy(),
        "site_screening_id":(
            manifest.latitude.iloc[:706].round(5).astype(str)+"|"+
            manifest.longitude.iloc[:706].round(5).astype(str)
        ).to_numpy(),
        "spatial_group":manifest.spatial_group.iloc[:706].astype(str).to_numpy(),
        "li_ppm":manifest.Li_icpms.iloc[:706].to_numpy(float),
        "y_true":label,
    }
    rows=[{"algorithm":PREVIOUS_ARM,"method":"release_protocol_retrained_old_706",
           **metrics(label,old.score)}]
    paired=[pd.DataFrame({**common,"arm":PREVIOUS_ARM,"score":old.score.to_numpy(float)})]
    for name,value in sorted(scores.items()):
        rows.append({"algorithm":name,"method":"new_competition_783",
                     **metrics(label,value[:706])})
        paired.append(pd.DataFrame({**common,"arm":name,"score":value[:706]}))
    pd.DataFrame(rows).to_csv(output/"old_release_protocol_comparison_706.csv",index=False)
    common_preds=pd.concat(paired,ignore_index=True)
    common_preds.to_csv(output/"old_release_protocol_paired_706.csv",index=False)
    interval=[]
    for name in sorted(scores):
        sample=common_preds[common_preds.arm.isin([PREVIOUS_ARM,name])]
        interval.append({"algorithm":name,
                         **bootstrap_group_delta(
                             sample,PREVIOUS_ARM,name,repeats=320,seed=20261112)})
    pd.DataFrame(interval).to_csv(output/"old_release_protocol_intervals_706.csv",
                                  index=False)
    triples,cases=ranking_evaluation(common_preds,triples_per_fold=250,seed=42)
    save_json(output/"old_release_protocol_three_zone.json",triples)
    cases.to_csv(output/"old_release_protocol_three_zone_cases.csv",index=False)
    result={
        "state":"complete","reference":PREVIOUS_ARM,
        "prior_fitted_release_artifact_used_as_test_predictor":False,
        "prior_release_protocol_retrained":True,
        "fixed_old_release_Li_cutoff_ppm":cutoff,
        "historical_heldout":706,"same_test_cases":True,"same_folds":True,
        "old_training_706_historical":True,
        "new_training_includes_77_provisional_georoc":True,
        "train_sets_and_fold_thresholds_are_not_identical":True,
        "exploratory_reused_development_data":True,
        "not_independent_external_validation":True,
        "algorithms":sorted(scores),
        "published_release_metrics_not_paired_or_directly_comparable":meta.get("metrics",{}),
        "rank_three_zones":{k:v["top1_accuracy"] for k,v in triples["arms"].items()},
    }
    save_json(output/"old_release_protocol_comparison_summary.json",result)
    display=pd.DataFrame(rows)
    pretty=["# Comparacion con protocolo del release M2 anterior","",
            "**Referencia reconstruida**, no evaluacion del mismo artefacto publicado.",
            "Los 706 casos y folds historicos son identicos, con corte Li fijo de "+
            str(cutoff)+" ppm. Los entrenamientos NO usan igual numero de muestras:",
            "el antiguo usa 706 historicas y el nuevo incluye 77 GEOROC provisionales.",
            "Los datos se reutilizaron para desarrollo; NO son test externo.","",
            "| Algoritmo | AUC | AP | Brier |",
            "|---|---:|---:|---:|"]
    for row in display.itertuples(index=False):
        pretty.append(f"| {row.algorithm} | {row.roc_auc:.4f} | {row.average_precision:.4f} | {row.brier:.4f} |")
    pretty+=["","Consultar old_release_protocol_intervals_706.csv y old_release_protocol_three_zone.json.",
             "Las metricas publicadas en septiembre estan en la metadata, pero no son comparables directamente."]
    (output/"OLD_RELEASE_PROTOCOL_COMPARISON.md").write_text(
        "\n".join(pretty)+"\n",encoding="utf-8")
    return result
