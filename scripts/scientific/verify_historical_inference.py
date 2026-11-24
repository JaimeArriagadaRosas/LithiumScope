"""Verify trusted local historical bundles by read-only copying, never activate or train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_1.prediction.predictor import predict_model_1_frame
from lithiumscope.model_2.prediction.predictor import score_model_2_image
from lithiumscope.prediction.analysis.decision import calculate_decision

RUN="demonstration_20261113_212844_m0300"
EXPECTED={
    "model_1":("training_20261106_111653_m0300","a09b5f0c2018f68ae567bf89f0b4f36c4d88dbd55bc7c7cb665d2140dc2f4a77"),
    "model_2":("training_20260920_195933_m0300","e41b275d5c0dfff6354e1f21b56d3cf1f4c9751abdb9415e50a571bdca14a45c")}
CASE_SHA="51b41d1f643ad7c0a05e64df6726a8f301169f4f03fdd9cad92604559842188a"


def verified_copy(source,target,records,expected=None):
    source,target=Path(source),Path(target)
    original=file_sha256(source)
    if expected and original!=expected:
        raise ValueError(f"Frozen source differs: {source.name}")
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    if file_sha256(target)!=original or file_sha256(source)!=original:
        raise ValueError(f"Source/copy drift: {source.name}")
    records[str(source)]=original
    return target


def run(old,output,*,trust_local_bundles=False):
    old,output=Path(old).resolve(),Path(output).resolve()
    if not trust_local_bundles:
        raise ValueError("Only load explicitly trusted local joblib artifacts")
    if old==PROJECT_ROOT.resolve() or output.is_relative_to(old) or not output.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Use separate historical source and diagnostic output under current clone")
    output.mkdir(parents=True,exist_ok=False)
    records={}
    archive=old/"results/predictions"/RUN
    copies=output/"frozen_evidence"
    manifest_file=verified_copy(archive/"prediction_manifest.json",copies/"prediction_manifest.json",records)
    manifest=json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest["run_id"]!=RUN or manifest["inputs"]["demonstration"]["cases_sha256"]!=CASE_SHA:
        raise ValueError("Historical manifest identity differs")
    bundles={}
    identities={}
    for model,(run_id,sha) in EXPECTED.items():
        identity=manifest["models"][model]
        if identity["run_id"]!=run_id or identity["model_sha256"]!=sha:
            raise ValueError("Historical model identity differs")
        source=old/"models"/model/"trained"/run_id
        meta=verified_copy(source/"metadata.json",copies/model/"metadata.json",records)
        metadata=json.loads(meta.read_text(encoding="utf-8"))
        if metadata["model_sha256"]!=sha or metadata["dataset_sha256"]!=identity["dataset_sha256"]:
            raise ValueError("Historical metadata mismatch")
        binary=verified_copy(source/"model.joblib",copies/model/"model.joblib",records,sha)
        bundles[model]=joblib.load(binary)
        identities[model]={"run_id":run_id,"sha256":sha,"algorithm":bundles[model]["algorithm"]}
    source_cases=verified_copy(old/"examples/integrated_demo/cases.csv",copies/"source_cases.csv",records,CASE_SHA)
    cases_file=verified_copy(archive/"inputs/demonstration_cases.csv",copies/"cases.csv",records)
    cases=pd.read_csv(cases_file)
    pd.testing.assert_frame_equal(pd.read_csv(source_cases),cases,check_exact=False,rtol=0,atol=1e-10)
    if len(cases)!=10 or not cases.case_id.is_unique:
        raise ValueError("Unexpected demonstration cases")
    archived={}
    for name in EXPECTED:
        path=verified_copy(archive/name/"predictions.csv",copies/name/"predictions.csv",records)
        archived[name]=pd.read_csv(path)
        if set(archived[name].case_id)!=set(cases.case_id) or not archived[name].case_id.is_unique:
            raise ValueError("Archived predictions do not match cases")
    one,diag=predict_model_1_frame(cases,bundle=bundles["model_1"])
    outputs=[]
    for row in cases.itertuples(index=False):
        expected=archived["model_2"].set_index("case_id").loc[row.case_id]
        image=old/"data/interim/model_2/integrated_demo"/f"{row.case_id}.npy"
        path=verified_copy(image,copies/"imagery"/image.name,records)
        side=verified_copy(image.with_suffix(".json"),path.with_suffix(".json"),records)
        meta=json.loads(side.read_text(encoding="utf-8"))
        if meta["scene_id"]!=expected.sentinel_scene_id or not np.allclose(
            [meta["latitude"],meta["longitude"]],[row.Latitude,row.Longitude],rtol=0,atol=1e-8):
            raise ValueError("Historical scene/coordinate provenance differs")
        payload,detail=score_model_2_image(path,bundle=bundles["model_2"],allow_acquisition=False)
        outputs.append({"case_id":row.case_id,**payload,"input_contract_status":detail.get("input_contract_status"),
                        "image_sha256":file_sha256(path),"scene_id":meta["scene_id"]})
    two=pd.DataFrame(outputs)
    comparisons=[]
    for model,predictions,column in (("model_1",one,"Li_icpms_predicted"),("model_2",two,"prospectivity_score")):
        left=archived[model][["case_id",column,"applicability_warning","out_of_training_range_fraction"]]
        current=predictions[["case_id",column,"applicability_warning","out_of_training_range_fraction"]]
        joined=left.merge(current,on="case_id",validate="one_to_one",suffixes=("_archived","_current"))
        joined["absolute_delta"]=(joined[column+"_current"]-joined[column+"_archived"]).abs()
        joined["warning_equal"]=joined.applicability_warning_archived.eq(joined.applicability_warning_current)
        joined["ood_fraction_delta"]=(joined.out_of_training_range_fraction_current-joined.out_of_training_range_fraction_archived).abs()
        joined.to_csv(output/(model+"_comparison.csv"),index=False)
        comparisons.append({"model":model,"rows":len(joined),"finite_predictions":int(joined[column+"_current"].notna().sum()),
                            "maximum_absolute_delta":float(joined.absolute_delta.max()) if joined.absolute_delta.notna().any() else None,
                            "matching_warnings":int(joined.warning_equal.sum()),
                            "prediction_parity":bool(np.allclose(joined[column+"_archived"],joined[column+"_current"],rtol=0,atol=1e-9)),
                            "ood_parity":bool(np.allclose(joined.out_of_training_range_fraction_archived,joined.out_of_training_range_fraction_current,rtol=0,atol=1e-12))})
    paired=one[["case_id","Li_icpms","Li_icpms_predicted","Li_icpms_interval_low_q90","applicability_warning"]].rename(
        columns={"applicability_warning":"model_1_applicability_warning"}).merge(
        two[["case_id","prospectivity_score","applicability_warning"]].rename(columns={"applicability_warning":"model_2_applicability_warning"}),
        on="case_id",validate="one_to_one")
    decision=calculate_decision(paired)
    paired.to_csv(output/"paired_cases.csv",index=False)
    decision.ranked.to_csv(output/"decision_ranked.csv",index=False)
    if any(file_sha256(Path(name))!=digest for name,digest in records.items()):
        raise ValueError("Historical source changed during audit")
    summary={"scope":"Ten known demonstration cases only; compatibility, not independent evaluation",
             "models":identities,"comparisons":comparisons,"decision_status":decision.status,
             "decision_winner":decision.winner_id,"decision_reason":decision.reason,
             "selected_m1":decision.selected.case_id.tolist() if "case_id" in decision.selected else [],
             "historical_sources_verified_unchanged":True,"source_sha256":records,
             "new_training":False,"activation":False,"pdf_rendered":False,"m1_diagnostics":diag}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"source_sha256","m1_diagnostics"}},indent=2,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trust-local-bundles",action="store_true")
    args=parser.parse_args()
    run(args.historical_root,args.output,trust_local_bundles=args.trust_local_bundles)
