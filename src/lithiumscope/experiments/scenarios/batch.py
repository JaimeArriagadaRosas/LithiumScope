"""Batch simulations and aggregate diagnostics; all models are read-only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from lithiumscope.experiments.scenarios.scenarios import catalog, perturb
from lithiumscope.model_1.prediction.predictor import predict_model_1_frame
from lithiumscope.experiments.scenarios.imagery import evaluate_model_2_image
from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery


def run_m1_batch(bundle: dict, frame: pd.DataFrame, destination: Path,
                 seeds=range(5)) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    truth = pd.to_numeric(frame["Li_icpms"], errors="coerce").to_numpy()
    base, _ = predict_model_1_frame(frame, bundle=bundle)
    baseline = base["Li_icpms_predicted"].to_numpy()
    records = []
    for seed in seeds:
        for scenario in catalog(seed):
            result, diagnostics = predict_model_1_frame(perturb(frame, scenario),bundle=bundle)
            predictions = result["Li_icpms_predicted"].to_numpy()
            mask = np.isfinite(truth) & np.isfinite(predictions)
            diff = predictions[mask] - truth[mask]
            records.append({"model":"model_1","seed":seed,"scenario":scenario.name,
                "family":scenario.kind,"intensity":scenario.intensity,"samples":int(mask.sum()),
                "rmse_ppm":float(np.sqrt(np.mean(diff**2))) if mask.any() else np.nan,
                "mae_ppm":float(np.mean(np.abs(diff))) if mask.any() else np.nan,
                "prediction_shift_ppm":float(np.mean(np.abs(predictions-baseline))),
                "ood_rate":diagnostics["out_of_domain_rows"]/max(len(frame),1)})
    path = destination / "batch_m1.csv"
    pd.DataFrame(records).to_csv(path,index=False)
    return path


def run_m2_batch(bundle: dict, manifest: Path, destination: Path,
                 *, accepted_only: bool = True) -> Path:
    frame = pd.read_csv(manifest)
    destination.mkdir(parents=True,exist_ok=True)
    outputs = []
    for item in frame.to_dict("records"):
        if accepted_only and item.get("status")!="ACCEPT":
            continue
        path = Path(item["path"])
        if not path.is_file():
            continue
        result = evaluate_model_2_image(bundle,load_imagery(path),
                                       destination / "per_image" / item["image_id"])
        rows = pd.read_csv(result)
        rows.insert(0,"image_id",item["image_id"])
        rows.insert(1,"scene_id",item["scene_id"])
        rows.insert(2,"year",item["year"])
        outputs.append(rows)
    output = destination / "batch_m2.csv"
    if outputs:
        pd.concat(outputs,ignore_index=True).to_csv(output,index=False)
    else:
        pd.DataFrame(columns=["image_id","scene_id","year","scenario","status","score"]).to_csv(output,index=False)
    return output


def summarize(output: Path, destination: Path) -> Path:
    table = pd.read_csv(output)
    destination.mkdir(parents=True,exist_ok=True)
    if table.empty:
        summary={"status":"NO_ACCEPTED_INPUTS","rows":0}
    else:
        summary={"status":"COMPLETE","rows":len(table),
            "scenarios":int(table["scenario"].nunique()),
            "images":int(table["image_id"].nunique()) if "image_id" in table else None,
            "notes":"Stress/robustness metrics only; cannot infer independent predictive accuracy."}
        for key in ("rmse_ppm","mae_ppm","ood_rate","absolute_score_shift"):
            if key in table:
                summary[f"mean_{key}"]=float(table[key].mean())
        if "status" in table:
            summary["status_counts"]=table["status"].value_counts().to_dict()
    target=destination/"summary.json"
    target.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return target
