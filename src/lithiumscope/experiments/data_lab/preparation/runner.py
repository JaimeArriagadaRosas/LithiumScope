"""Approved final M1 preparation, without fitting predictive models."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import DATA_DIR, RESULTS_DIR, PROJECT_ROOT
from lithiumscope.core.logging.session import get_logger
from lithiumscope.experiments.generalization.prepared import prepare_version, load_snapshot
from lithiumscope.results.atomic_exports import write_json, write_table
from .evidence import BASE_SHA, INVENTORY_SHA, load_checkpoint, checked_file
from .dispositions import apply_decisions
from .reconciliation import reconcile, candidate_preflight
from .recovery import recover, ANDEAN, independent_links
from .historical_recovery import review_originals
from .fold_preparation import prepare_folds

logger = get_logger("experiments.data_lab.preparation")


def stage(number, title):
    print(f"\n[{number}/6] {title}", flush=True)
    logger.info("M1 final preparation stage=%d title=%s",number,title)


def _run_directory(root):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / "model_1/final_data_preparation" / f"preparation_{stamp}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True,exist_ok=False)
    return directory


def run(source, *, root=None, research_root=None):
    root = Path(root) if root is not None else RESULTS_DIR
    research_root = Path(research_root) if research_root is not None else DATA_DIR / "raw/model_1/research"
    source = Path(source)
    print("="*72+"\nPREPARACIÓN Y TRATAMIENTO | MODELO 1\n"+"="*72)
    cfg = load_config("model_1")
    stage(1,"Verificación de evidencias EDA y cohorte congelada")
    eda,eda_manifest,profiles,issues = load_checkpoint(source,cfg,root,research_root)
    snapshot,_ = prepare_version("model_1",source,run_eda=False)
    data = load_snapshot(snapshot,snapshot.name)
    x,y,groups,_,_,numeric,categorical=data
    if len(x)!=eda_manifest["rows_prepared"] or len(x.columns)!=eda_manifest["features"]:
        raise ValueError("Prepared cohort differs from verified EDA")
    base = pd.read_csv(source,low_memory=False,encoding=eda_manifest["tabular_source"]["encoding"],
                       sep=eda_manifest["tabular_source"]["delimiter"])
    if len(base)!=eda_manifest["rows_raw"]:
        raise ValueError("Canonical raw row count differs")
    qc = pd.read_csv(eda / "tables/qc_recovery_review.csv")
    directory = _run_directory(root)
    failures = {}
    try:
        print(f"    Cohorte: {len(x)} filas | {len(numeric)+len(categorical)} variables | {groups.nunique()} grupos")
        print(f"    Fuentes con evidencia íntegra: {len(profiles)}/6")
        stage(2,"Decisiones de fuentes adicionales")
        decisions = apply_decisions(profiles,issues,root/"model_1/source_catalog")
        write_json(directory/"source_decisions.json",decisions)
        for row in decisions:
            print(f"    {row['source']:<30} {row['state']}")
            if row["state"] not in {"retired","already_retired","restricted"}:
                failures[row["source"]]=row.get("issue",row["state"])
        stage(3,"Recuperación puntual y reconciliación GEOROC")
        candidates=pd.DataFrame()
        try:
            counts,differences,candidates = reconcile(profiles,base,root,
                cfg["data"]["quality_min"],cfg["data"]["quality_max"])
            write_table(directory/"georoc_reconciliation.csv",counts)
            write_table(directory/"georoc_membership_differences.csv",differences)
            write_table(directory/"georoc_review_candidates.csv",candidates)
            print(f"    Candidatas para revisión individual: {len(candidates)} (no admitidas)")
        except (OSError,ValueError,KeyError,TypeError) as exc:
            failures["reconciliation"]=str(exc)
            logger.exception("GEOROC reconciliation blocked")
            print(f"    [BLOQUEADO] Reconciliación: {exc}")
        proposals,publications,links = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        if ANDEAN in profiles:
            try:
                links,publications,proposals = recover(base,qc,profiles[ANDEAN])
                write_table(directory/"recovery_identity_links.csv",links)
                write_table(directory/"publication_recovery_review.csv",publications)
                write_table(directory/"oxide_recovery_review.csv",proposals)
                print(f"    Enlaces únicos sin Li: {int(links.unique_link.sum())}")
                print(f"    Referencias para revisión: {len(publications)} | mediciones propuestas: {len(proposals)}")
            except (OSError,ValueError,KeyError,TypeError) as exc:
                failures["recovery"]=str(exc)
                logger.exception("Original recovery blocked")
                print(f"    [BLOQUEADO] Recuperación: {exc}")
        else:
            failures["recovery"]="Verified Andean profile unavailable"
        # Search original historical files only for the identified QC gaps.
        original_proposals, original_review = review_originals(base,qc,cfg,PROJECT_ROOT)
        write_table(directory/"historical_original_review.csv",original_review)
        proposals=pd.concat([proposals,original_proposals],ignore_index=True)
        if len(proposals):
            # Preserve contradictory values for review; never pick one silently.
            proposals["conflicting_values"] = proposals.groupby(["base_position","oxide"]).proposed_value_wt_pct.transform("nunique").gt(1)
        write_table(directory/"oxide_recovery_review.csv",proposals)
        unresolved = qc.loc[qc.reason.eq("missing_or_unparseable")].copy()
        proposed = set(proposals.get("base_position",pd.Series(dtype=str)).astype(str))
        unresolved["measurement_proposal_available"] = unresolved.row_index.astype(str).isin(proposed)
        unresolved["approved_for_training"] = False
        write_table(directory/"qc_recovery_disposition.csv",unresolved)
        if len(original_review) and original_review.state.eq("blocked").any():
            failures["historical_originals"]="Some original historical partitions or lineage unavailable"
        if "macusani_2025" in profiles:
            # Independent identity diagnostic only: no Macusani rows or values admitted.
            mac=pd.read_csv(profiles["macusani_2025"]["path"]/"tables/review_rows.csv",low_memory=False)
            mac_links=independent_links(base,mac)
            mac_links["disposition"]="quarantine"
            write_table(directory/"macusani_identity_review.csv",mac_links)
        print(f"    Propuestas de medición (todas pendientes): {len(proposals)}")
        stage(4,"Contratos de predictores")
        print("    Control | sin categorías | ILR de óxidos | indicadores de ausencia")
        print("    No se repiten balanceos sintéticos ni campañas de transferencia cerradas.")
        write_json(directory/"treatment_decisions.json",{
            "control":"reference", "new_feature_variants":["without_categories","ilr_oxides","chemistry_missing_indicators"],
            "prior_policies_not_repeated":["SMOTER","rare/group bootstrap","density and source/group weighting",
                "broad GEOROC transfer","Chen C1/C2/D","historical log-ratios","KNN/lithology imputation",
                "regional-context ablation","TabNet missing mask"],
            "reason":"Historical campaigns archived; final preparation isolates non-equivalent feature hypotheses.",
            "conditional_variants":{"recovered_measurements":"requires original identity, units and analytical basis approval",
                "real_expansion":"requires certified new identities, metadata, TRAIN compatibility and explicit review",
                "combined":"requires individual predictive evaluation and approved composition"},
            "model_fits":0,"production_changed":False})
        stage(5,"Preparación exterior e interior exclusivamente TRAIN")
        plan,protocol,table,states,splits,coverage=prepare_folds(data,cfg,eda_manifest["development_protocol"])
        write_table(directory/"treatments.csv",table)
        write_json(directory/"transform_contracts.json",states)
        write_json(directory/"partitions.json",splits)
        write_table(directory/"train_concentration_coverage.csv",coverage)
        try:
            historical=base.loc[x.index.astype(int)]
            screening=candidate_preflight(candidates,x,groups,plan,historical=historical)
            write_table(directory/"candidate_fold_preflight.csv",screening)
            if len(screening):
                counts=screening.groupby(["source","outer_fold","inner_fold"],dropna=False).agg(
                    review_rows=("review_id","size"),supported=("chemistry_nn_supported","sum"),
                    above_696=("above_696","sum"),reserved_group=("group_reserved","sum"),
                    reserved_publication=("known_reserved_publication_overlap","sum"),
                    approved_rows=("approved_for_training","sum")).reset_index()
                write_table(directory/"candidate_fold_summary.csv",counts)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            failures["candidate_preflight"]=str(exc)
            logger.exception("Candidate preflight blocked")
            print(f"    [BLOQUEADO] Soporte de candidatas: {exc}")
        for arm, subset in table.groupby("arm",sort=False):
            ready=subset.loc[subset.state.eq("ready")]
            detail = "" if ready.empty else f" | variables {int(ready.features_after.min())}-{int(ready.features_after.max())}"
            print(f"    {arm:<30} {len(ready)}/{len(subset)} listas{detail}")
        stage(6,"Validación y exportación")
        checked_file(source,BASE_SHA)
        # Reverify the exact snapshot after processing and never overwrite it.
        load_snapshot(snapshot,snapshot.name)
        if not table.loc[table.arm.eq("reference"),"state"].eq("ready").all():
            raise ValueError("Canonical control preflight blocked")
        if not table.state.eq("ready").all():
            failures["variants"]="One or more fold-local contracts blocked"
        summary={"rows":len(x),"features":len(x.columns),"groups":int(groups.nunique()),
                 "source_retirements":sum(r["state"] in {"retired","already_retired"} for r in decisions),
                 "candidate_review_rows":len(candidates),"publication_proposals":len(publications),
                 "measurement_proposals":len(proposals),"fold_partitions":sum(1+len(i) for _,_,i in plan),
                 "variant_checks":len(table),"ready":int(table.state.eq("ready").sum()),
                 "blocked":int(table.state.eq("blocked").sum()),"new_rows_approved":0,"model_fits":0}
        from importlib.metadata import version
        code={module.name:file_sha256(module) for module in sorted(Path(__file__).parent.glob("*.py"))}
        manifest={"schema_version":1,"state":"partial" if failures else "complete",
            "scope":"M1_final_train_only_preparation_not_predictive_evaluation",
            "source_sha256":BASE_SHA,"eda_run_id":eda.name,
            "eda_manifest_sha256":file_sha256(eda/"manifest.json"),
            "source_inventory_sha256":INVENTORY_SHA,
            "prepared_identity":snapshot.name,"prepared_manifest_sha256":file_sha256(snapshot/"manifest.json"),
            "development_protocol":protocol,"summary":summary,"failures":failures,
            "source_unchanged":True,"predictive_improvement_verified":False,"production_changed":False,
            "evaluation_authorized":False,"implementation_sha256":code,
            "versions":{name:version(name) for name in ("numpy","pandas","scikit-learn")},
            "artifacts":{path.name:file_sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}}
        path=write_json(directory/"manifest.json",manifest)
        for name,digest in manifest["artifacts"].items():
            checked_file(directory/name,digest)
        print("="*72+"\nRESUMEN FINAL M1\n"+"="*72)
        print(f"    Fuentes retiradas: {summary['source_retirements']} | candidatas revisadas: {len(candidates)}")
        print(f"    Contratos listos: {summary['ready']}/{summary['variant_checks']}")
        print("    Nuevas filas aprobadas: 0 | Modelos entrenados: 0")
        print(f"    Evidencias: {directory}")
        print("    La evaluación predictiva posterior requiere revisar estas evidencias.")
        return manifest["state"],{"preparation_path":str(directory),"preparation_manifest":manifest,
            "manifest_sha256":file_sha256(path),"eda_path":str(eda),"prepared_path":str(snapshot),
            "source_decisions":decisions,"model_fits":0}
    except Exception as exc:
        logger.exception("M1 final preparation failed")
        write_json(directory/"failure.json",{"state":"failed","error":str(exc),"model_fits":0})
        raise
