"""Read-only parity of the two known historical bundles and archived ten-case demo."""
from __future__ import annotations

import json
from pathlib import Path

from lithiumscope.core.hashing import file_sha256
from lithiumscope.results.atomic_exports import write_json

RUN="demonstration_20261113_212844_m0300"
EXPECTED={
    "model_1":("training_20261106_111653_m0300","a09b5f0c2018f68ae567bf89f0b4f36c4d88dbd55bc7c7cb665d2140dc2f4a77"),
    "model_2":("training_20260920_195933_m0300","e41b275d5c0dfff6354e1f21b56d3cf1f4c9751abdb9415e50a571bdca14a45c")}
CASE_SHA="51b41d1f643ad7c0a05e64df6726a8f301169f4f03fdd9cad92604559842188a"


def _required(root):
    archive=root/"results/predictions"/RUN
    paths=[root/"examples/integrated_demo/cases.csv",archive/"prediction_manifest.json",
           archive/"inputs/demonstration_cases.csv"]
    for group,(run_id,_) in EXPECTED.items():
        directory=root/"models"/group/"trained"/run_id
        paths.extend([directory/"model.joblib",directory/"metadata.json",archive/group/"predictions.csv"])
    return paths


def verify(root, output):
    """Never deserialize an unknown hash; absence is blocked, not an invented pass."""
    root,output=Path(root).resolve(),Path(output)
    if not output.resolve().is_relative_to(root):
        raise ValueError("Historical parity output must remain in workspace")
    archive=root/"results/predictions"/RUN
    inputs=_required(root)
    report={"state":"BLOCKED","scope":"historical ten-case inference parity; not external validation",
            "training":False,"activation":False,"acquisition":False,"comparisons":{}}
    missing=[p.relative_to(root).as_posix() for p in inputs if not p.is_file()]
    if missing:
        report.update(reason="historical_artifacts_absent",missing=missing)
        write_json(output,report)
        return report
    import pandas as pd
    import numpy as np
    hashes={path:file_sha256(path) for path in inputs}
    manifest=json.loads((archive/"prediction_manifest.json").read_text(encoding="utf-8"))
    if (hashes[root/"examples/integrated_demo/cases.csv"]!=CASE_SHA
            or manifest["run_id"]!=RUN or manifest["inputs"]["demonstration"]["cases_sha256"]!=CASE_SHA):
        report.update(state="ERROR",reason="historical_manifest_or_case_identity_mismatch")
        write_json(output,report)
        return report
    cases=pd.read_csv(archive/"inputs/demonstration_cases.csv")
    pd.testing.assert_frame_equal(pd.read_csv(root/"examples/integrated_demo/cases.csv"),cases,
                                  check_exact=False,rtol=0,atol=1e-10)
    if len(cases)!=10 or not cases.case_id.is_unique:
        raise ValueError("Historical demonstration cohort mismatch")
    for group,(run_id,digest) in EXPECTED.items():
        directory=root/"models"/group/"trained"/run_id
        identity=manifest["models"][group]
        metadata=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
        if (hashes[directory/"model.joblib"]!=digest or metadata["model_sha256"]!=digest
                or identity["run_id"]!=run_id or identity["model_sha256"]!=digest
                or metadata["dataset_sha256"]!=identity["dataset_sha256"]):
            report.update(state="ERROR",reason="historical_bundle_identity_mismatch",model=group)
            write_json(output,report)
            return report
    archived={group:pd.read_csv(archive/group/"predictions.csv") for group in EXPECTED}
    for frame in archived.values():
        if not frame.case_id.is_unique or set(frame.case_id)!=set(cases.case_id):
            raise ValueError("Historical prediction cohort mismatch")
    images={}
    for row in cases.itertuples(index=False):
        name=str(row.case_id)
        if Path(name).name!=name or name in {".",".."}:
            raise ValueError("Invalid case identity")
        path=root/"data/interim/model_2/integrated_demo"/f"{name}.npy"
        side=path.with_suffix(".json")
        if not path.is_file() or not side.is_file():
            report.update(reason="historical_imagery_absent",missing=[p.relative_to(root).as_posix() for p in (path,side) if not p.is_file()])
            write_json(output,report)
            return report
        for p in (path,side):
            hashes[p]=file_sha256(p)
        meta=json.loads(side.read_text(encoding="utf-8"))
        old=archived["model_2"].set_index("case_id").loc[row.case_id]
        if (meta["scene_id"]!=old.sentinel_scene_id or not np.allclose(
                [meta["latitude"],meta["longitude"]],[row.Latitude,row.Longitude],rtol=0,atol=1e-8)):
            raise ValueError("Historical scene identity mismatch")
        if meta.get("image_sha256") and meta["image_sha256"]!=hashes[path]:
            raise ValueError("Historical imagery hash mismatch")
        images[row.case_id]=path
    from lithiumscope.persistence.model_deserialization import load_model_artifact
    from lithiumscope.model_1.prediction.predictor import predict_model_1_frame
    from lithiumscope.model_2.prediction.predictor import score_model_2_image
    bundles={}
    for group,(run_id,digest) in EXPECTED.items():
        path=root/"models"/group/"trained"/run_id/"model.joblib"
        # Read once, then deserialize exactly the verified bytes (not a reopened path).
        from io import BytesIO
        from hashlib import sha256
        payload=path.read_bytes()
        if sha256(payload).hexdigest()!=digest:
            raise ValueError("Historical model bytes changed")
        bundles[group]=load_model_artifact(BytesIO(payload), model_group=group)
    one,_=predict_model_1_frame(cases,bundle=bundles["model_1"])
    two=[]
    for sid,path in images.items():
        result,_=score_model_2_image(path,bundle=bundles["model_2"],allow_acquisition=False)
        two.append({"case_id":sid,**result})
    for group,frame,column in (("model_1",one,"Li_icpms_predicted"),("model_2",pd.DataFrame(two),"prospectivity_score")):
        keys=["case_id",column,"applicability_warning","out_of_training_range_fraction"]
        paired=archived[group][keys].merge(frame[keys],on="case_id",validate="one_to_one",suffixes=("_old","_new"))
        a,b=paired[column+"_old"].to_numpy(float),paired[column+"_new"].to_numpy(float)
        valid=len(paired)==10 and np.isfinite(a).all() and np.isfinite(b).all()
        report["comparisons"][group]={"rows":len(paired),"prediction_parity":bool(valid and np.allclose(a,b,rtol=0,atol=1e-9)),
            "maximum_absolute_delta":float(np.max(np.abs(a-b))) if valid else None,
            "warning_parity":bool(len(paired)==10 and paired.applicability_warning_old.eq(paired.applicability_warning_new).all()),
            "ood_parity":bool(len(paired)==10 and np.allclose(paired.out_of_training_range_fraction_old,paired.out_of_training_range_fraction_new,rtol=0,atol=1e-12))}
    unchanged=all(file_sha256(path)==digest for path,digest in hashes.items())
    report["historical_sources_unchanged"]=unchanged
    report["state"]="OK" if unchanged and all(
        x["prediction_parity"] and x["warning_parity"] and x["ood_parity"] for x in report["comparisons"].values()) else "ERROR"
    write_json(output,report)
    return report
