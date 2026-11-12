"""Post-competition scientific reports: all offline and recoverable.

Never actually runs a model fit or modifies stored candidate artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.training import postprocess as pp
from lithiumscope.model_2.training.competition import _excel_sheets


def create_manifest(path:Path):
    n=783
    # First 706 are the historical reference used to match old release.
    ids=[f"S{i:03d}" for i in range(n)]
    df=pd.DataFrame({
        "sample_id":ids,
        "Li_icpms":np.where(np.arange(n)%4==0,40.0,13.0),
        "latitude":-15.0-np.arange(n)*.001,
        "longitude":-70.0-np.arange(n)*.001,
        "spatial_group":[f"g{i//150}" for i in range(n)],
    })
    df.to_csv(path,index=False)
    return df


def create_oof(tables:Path,manifest:pd.DataFrame,*,mismatch:bool=False):
    tables.mkdir(parents=True,exist_ok=True)
    n=len(manifest)
    fold=1+(np.arange(n)%5)
    scores=np.clip(.15+.01*(np.arange(n)%17),.05,.9)
    labels=(manifest.Li_icpms.to_numpy(float)>21.6).astype(int)
    for name in ("random_forest","catboost"):
        records=pd.DataFrame({
            "y_true":labels,
            "li_ppm":manifest.Li_icpms.to_numpy(float),
            "outer_fold":fold,
            "prospectivity_score":scores if name=="random_forest" else 1-scores,
        })
        if mismatch and name=="catboost":
            records.loc[0,"outer_fold"]=3
        records.to_csv(tables/f"oof_predictions_{name}.csv",index=False)


def test_ambiguous_oof_is_rejected_before_reporting(tmp_path):
    manifest=create_manifest(tmp_path/"manifest.csv")
    tables=tmp_path/"tables"
    create_oof(tables,manifest,mismatch=True)
    with pytest.raises(ValueError,match="disagree"):
        pp.load_predictions(manifest,tables)


def test_recovery_status_is_persisted_and_reports_retry_without_retraining(
    tmp_path,monkeypatch
):
    manifest=create_manifest(tmp_path/"manifest.csv")
    tables=tmp_path/"tables";create_oof(tables,manifest)
    root=tmp_path/"workspace"
    output=tmp_path/"output"
    requested=[]
    fail_once=[True]
    def old(*args):
        requested.append("old")
        if fail_once[0]:
            raise ValueError("Injected missing source")
        return {"state":"complete"}
    def science(*args):
        requested.append("science")
        return {"state":"complete"}
    def triple(*args):
        requested.append("triple")
        return {"state":"complete"}
    monkeypatch.setattr(pp,"fixed_old_release_comparison",old)
    monkeypatch.setattr(pp,"science_report",science)
    monkeypatch.setattr(pp,"three_zone_report",triple)
    one=pp.finalize_reports(tmp_path/"manifest.csv",tables,output,root=root)
    assert one["state"]=="partial_requires_recovery"
    assert one["reports"]["old_release_comparison"]["state"]=="requires_recovery"
    assert one["reports"]["scientific_diagnostics"]["state"]=="complete"
    assert one["reports"]["three_zone"]["state"]=="complete"
    assert "python -m lithiumscope.model_2.training.postprocess" in one["recovery_command"]
    assert (output/"EVIDENCE_REPORT_STATUS.md").is_file()
    fail_once[0]=False
    two=pp.finalize_reports(tmp_path/"manifest.csv",tables,output,root=root,strict=True)
    assert two["state"]=="complete"
    assert all(v["state"]=="complete" for v in two["reports"].values())
    assert len(requested)==6
    assert pp.get_json(output/"evidence_report_status.json")["runs_no_training"]


def test_release_comparator_rejects_mismatched_frozen_folds(tmp_path,monkeypatch):
    root=tmp_path
    old_dir=root/"results/model_2/runs/training_20261112_112450_m0300"
    old_dir.mkdir(parents=True)
    meta_dir=root/"models/model_2/trained/training_20260920_195933_m0300"
    meta_dir.mkdir(parents=True)
    (meta_dir/"metadata.json").write_text(json.dumps({
        "samples":706,"algorithm":"catboost","threshold_ppm":21.6
    }),encoding="utf-8")
    manifest=create_manifest(root/"manifest.csv")
    old=pd.DataFrame({
        "arm":[pp.PREVIOUS_ARM]*706,"test_index":range(706),
        "sample_id":manifest.sample_id.iloc[:706],
        "li_ppm":manifest.Li_icpms.iloc[:706],
        "fold":np.ones(706,int),
        "score":np.full(706,.33)
    })
    old.to_csv(old_dir/"all_reference_paired_oof.csv",index=False)
    from lithiumscope.core.hashing import file_sha256
    monkeypatch.setattr(pp,"EXPECTED_OLD_REFERENCE_SHA256",
                        file_sha256(old_dir/"all_reference_paired_oof.csv"))
    monkeypatch.setattr(pp,"EXPECTED_OLD_METADATA_SHA256",
                        file_sha256(meta_dir/"metadata.json"))
    with pytest.raises(ValueError,match="heldout folds"):
        pp.fixed_old_release_comparison(
            root,manifest,np.asarray([2]*783),
            {"catboost":np.full(783,.25)},root)


def test_three_zone_report_all_six_competitors_reproducible_without_model(
    tmp_path
):
    manifest=create_manifest(tmp_path/"manifest.csv")
    # Fold by contiguous independent site-groups, never split the same site.
    folds=1+(np.arange(783)//157)
    scores={}
    for k,algorithm in enumerate((
        "random_forest","extra_trees","hist_gradient_boosting",
        "xgboost","catboost","svm_rbf"
    )):
        scores[algorithm]=np.clip(
            .1+((np.arange(783)+k*3)%53)/60.,0.05,0.99)
    label=(manifest.Li_icpms.to_numpy(float)>=21.6).astype(int)
    result=pp.three_zone_report(tmp_path/"manifest.csv",folds,label,
                                scores,tmp_path)
    assert result["state"]=="complete"
    assert len(result["algorithms"])==6
    assert result["case_rows"]>2000
    assert (tmp_path/"three_zone_competition_summary.json").exists()
    assert set(result["summary"]["arms"])==set(scores)


def test_excel_export_includes_scientific_comparison_when_present(tmp_path):
    ranking=pd.DataFrame([{
        "algorithm":"catboost","status":"ok","roc_auc_mean":.6}])
    class Result:
        fold_table=pd.DataFrame({"fold":[1],"roc_auc":[.6]})
        probabilities=np.asarray([.2,.6])
    y=pd.Series([0,1])
    (tmp_path/"old_release_protocol_comparison_706.csv").write_text(
        "algorithm,roc_auc\nold,0.56\nnew,0.62\n",encoding="utf-8")
    (tmp_path/"scientific_stratified_metrics.csv").write_text(
        "algorithm,subset,n\ncatboost,all,783\n",encoding="utf-8")
    sheets=_excel_sheets(ranking,{"catboost":Result()},y,
                         (None,pd.DataFrame({"fold":[1]})),tmp_path)
    assert "release_706" in sheets
    assert "scientific_strata" in sheets
    assert "ranking" in sheets
    assert "pred_catboost" in sheets
