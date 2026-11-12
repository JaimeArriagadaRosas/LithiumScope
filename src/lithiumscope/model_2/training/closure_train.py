"""Separated train responsibility; preserve prior public API."""
from __future__ import annotations

from lithiumscope.model_2.training.closure import PROJECT_ROOT
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.model_2.training.closure import Path

def train(root:Path=PROJECT_ROOT, *,preflight_only=False,
          smoke=False,output:Path|None=None):
    from lithiumscope.model_2.training.closure import LIFECYCLE, OLD_DISCARDED_ID, OLD_RELEASE_ID, PARAMS, PROJECT_ROOT, Path, _evaluate, _model_from_params, define_outer_fold, finite_quality_filter, json, np, prepare, write_closure_report
    print("Modelo 2: verificando 783 hashes, 706 muestras historicas, "
          "5 folds y correspondencia de caracteristicas...",flush=True)
    info,contract,X,folds,qa,eligible,old_params,archived=prepare(root)
    print("Preflight OK: 690 filas de train elegibles, "
          "706 test originales, GEOROC provisional excluido.",flush=True)
    if preflight_only:
        if output is not None:
            output.mkdir(parents=True,exist_ok=True)
            (output/"preflight.json").write_text(
                json.dumps(info,indent=2,ensure_ascii=False,default=str),
                encoding="utf-8")
        return info
    if smoke:
        dest=output or root/"results/scientific_simulations/model2_manual_closure_smoke"
        dest.mkdir(parents=True,exist_ok=True)
        _,tr,test,threshold=define_outer_fold(folds[0],contract)
        tr=finite_quality_filter(tr,qa)[:125]
        test=test[:16]
        y=(contract.Li_icpms.iloc[tr].to_numpy(float)>=threshold).astype(int)
        if len(set(y))!=2:raise ValueError("Smoke subset lacks 2 classes")
        scored=_model_from_params(X.iloc[tr],y,X.iloc[test],PARAMS)
        report={"state":"smoke_only","candidate_created":False,
                "n_train":len(tr),"test":len(test),
                "finite_scores":bool(np.isfinite(scored).all()),
                "no_release":True}
        (dest/"smoke.json").write_text(json.dumps(report,indent=2),
                                       encoding="utf-8")
        return report
    from lithiumscope.core.run_context import RunContext
    from lithiumscope.core.states import RunState
    from lithiumscope.core.training_logs import register_training_run
    from lithiumscope.model_2.schema import feature_range_profile
    from lithiumscope.persistence.save_model import save_model_bundle
    from catboost import CatBoostClassifier
    context=RunContext.create("model_2",resume=False)
    out=output or context.root
    out.mkdir(parents=True,exist_ok=True)
    context.tracker.set_state(RunState.RUNNING)
    (out/"manual_closure_preflight.json").write_text(
        json.dumps(info,indent=2,ensure_ascii=False,default=str),
        encoding="utf-8")
    print("MODELO 2: entrenando prototipo experimental; NO RELEASE",flush=True)
    print("Calidad 95% SOLO TRAIN; los 706 TEST históricos se preservan.",flush=True)
    try:
        evaluated=_evaluate(root,out,contract,X,folds,qa,old_params,archived)
    except Exception as exc:
        context.tracker.set_state(RunState.FAILED,failure=str(exc))
        raise
    final_threshold=float(np.quantile(contract.Li_icpms.iloc[:706].to_numpy(float),.75))
    y=(contract.Li_icpms.iloc[eligible].to_numpy(float)>=final_threshold).astype(int)
    model=CatBoostClassifier(**PARAMS)
    model.fit(X.iloc[eligible],y)
    bundle={
        "estimator":model,"features":list(X.columns),
        "band_names":["red","green","blue","nir","swir16","swir22"],
        "normalize_per_band":False,
        "algorithm":"catboost",
        "threshold_ppm":final_threshold,"target_quantile":.75,
        "operating_threshold":.5,
        "operating_threshold_objective":"rank_only_fixed_threshold",
        "feature_extractor_version":"legacy_126_DN_median",
        "spectral_representation":"legacy_DN_median",
        "quality_min_joint_valid":.95,
        "experimental_only":True,
        "ranking_only_not_deposit_probability":True,
        "applicability_profile":feature_range_profile(X.iloc[eligible]),
    }
    meta={
        "model_group":"model_2","algorithm":"catboost",
        "experiment_state":LIFECYCLE,"not_for_release":True,
        "activation_forbidden":True,
        "previous_discarded_experiment":OLD_DISCARDED_ID,
        "previous_release_reference":OLD_RELEASE_ID,
        "historical_samples":706,"trained_eligible":len(eligible),
        "excluded_low_coverage_from_training":16,
        "heldout_unchanged":706,
        "georoc_provisional_training_count":0,
        "Li_cutoff_source":"historical_all_706_q75",
        "params":PARAMS,
        "spectral_representation":"legacy_DN_median_126",
        "evaluation_summary":evaluated,
        "scope":"three-zone ranking research; not release or economic deposit estimate",
    }
    model_path,metadata_path=save_model_bundle(
        "model_2","experimental_model2_corrected_quality95",
        bundle,meta,
        run_id=context.run_id,config_name="model_2",
        feature_schema={
            "features":list(X.columns),
            "representation":"legacy_DN_median_126",
            "quality_min_joint_valid":.95,
        },activate=False,
    )
    status_file=model_path.parent/"candidate_status.json"
    status=json.loads(status_file.read_text(encoding="utf-8"))
    status.update({
        "status":"experimental_awaiting_review",
        "not_for_release":True,
        "eligible_for_individual_publication":False,
        "candidate_discardable":True,
    })
    status_file.write_text(json.dumps(
        status,indent=2,ensure_ascii=False),encoding="utf-8")
    assessment=write_closure_report(
        out,info,evaluated,model_path=model_path)
    # Preserve standard metric-center access without claiming a six-model
    # competition: this is an explicitly labelled 2-protocol comparison.
    metrics_file=out/"metrics_comparison.csv"
    if metrics_file.is_file():
        from shutil import copyfile
        copyfile(metrics_file,context.tables/"experimental_protocol_comparison.csv")
    report={
        "state":"completed_manual_experimental_only",
        "model_path":str(model_path),
        "metadata_path":str(metadata_path),
        "run_dir":str(out),
        "active_release_untouched":True,
        "training_count":len(eligible),
        "test_count":706,"GEOROC_provisional_count":0,
        "new_release_created":False,
        "comparison":evaluated,
        "descriptive_internal_acceptance":assessment["passes_descriptive_internal_criteria"],
        "report_path":str(out/"INFORME_ENTRENAMIENTO_MODELO2_EXPERIMENTAL.md"),
    }
    (out/"manual_training_summary.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False,default=str),
        encoding="utf-8")
    context.tracker.set_state(
        RunState.COMPLETED,winner="experimental_catboost_quality95",
        release_gate_pass=False,experimental_only=True,
        primary_metric="roc_auc",
        primary_metric_value=float(evaluated["metrics"]["corrected_quality95"]["auc"]),
        manual_closure_accepted_internally=assessment["passes_descriptive_internal_criteria"],
        model_path=str(model_path))
    register_training_run("model_2",out)
    print("EXPERIMENTAL_M2_COMPLETE",str(out),flush=True)
    print("NO RELEASE; NO ACTIVE MODEL CHANGE",flush=True)
    return report
