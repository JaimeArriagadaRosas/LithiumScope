"""Inference-only scientific stress tests for LithiumScope M1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

GROUPS = {
    "oxides": ("SiO2","TiO2","Al2O3","Fe2O3","MnO","MgO","CaO","Na2O","K2O","P2O5"),
    "traces": ("Th_icpms","U_icpms","Rb_icpms","Cs_icpms","Nb_icpms","Ta_icpms","Pb_icpms","Ba_icpms","Sr_icpms","Zr_icpms","V_icpms","Hf_icpms"),
    "geography": ("Longitude","Latitude","Longitude (X)","Latitude (Y)"),
    "geology": ("Age (Ma)","Geologycal_age","Sample_type","Rock_type","Arc","Domain"),
}
PROTECTED = {"Li_icpms", "case_id", "Sample", "sample_id", "source_dataset", "source_sample"}


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    intensity: float
    seed: int = 42


def catalog(seed: int = 42) -> list[Scenario]:
    result = [Scenario("baseline", "baseline", 0, seed)]
    result.extend(Scenario(f"missing_{p:g}", "missing", p, seed) for p in (.05,.10,.25,.50,.75,1.0))
    result.extend(Scenario(f"group_{g}", "group", 1.0, seed) for g in GROUPS)
    result.extend(Scenario(f"noise_{p:g}", "noise", p, seed) for p in (.01,.05,.10,.25))
    result.extend(Scenario(f"extreme_{p:g}", "extreme", p, seed) for p in (1.5,3.0,10.0))
    return result


def perturb(frame: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    out = frame.copy(deep=True)
    rng = np.random.default_rng(scenario.seed)
    cols = [c for c in out if c not in PROTECTED]
    if scenario.kind == "baseline":
        return out
    if scenario.kind == "missing":
        out.loc[:,cols] = out[cols].mask(rng.random((len(out),len(cols))) < scenario.intensity)
    elif scenario.kind == "group":
        for col in GROUPS[scenario.name.removeprefix("group_")]:
            if col in out:
                out[col] = np.nan
    elif scenario.kind in ("noise","extreme"):
        candidates = [c for c in cols if c not in GROUPS["geography"] and pd.api.types.is_numeric_dtype(out[c])]
        for col in candidates:
            values = pd.to_numeric(out[col],errors="coerce")
            if scenario.kind == "noise":
                out[col] = values + rng.normal(0,scenario.intensity*max(float(values.std()),1e-9),len(out))
            else:
                out[col] = values * scenario.intensity
    else:
        raise ValueError(f"Unknown scenario kind: {scenario.kind}")
    return out


def evaluate_model_1(bundle: dict, frame: pd.DataFrame, destination: Path, seed: int = 42) -> Path:
    """Run fixed fitted bundle; never fit, alter or promote the model."""
    from lithiumscope.model_1.prediction.predictor import predict_model_1_frame
    destination.mkdir(parents=True,exist_ok=True)
    baseline, _ = predict_model_1_frame(frame, bundle=bundle)
    base = pd.to_numeric(baseline["Li_icpms_predicted"],errors="coerce").to_numpy()
    records = []
    for scenario in catalog(seed):
        predictions, diagnostics = predict_model_1_frame(perturb(frame,scenario),bundle=bundle)
        values = pd.to_numeric(predictions["Li_icpms_predicted"],errors="coerce").to_numpy()
        rec = {
            "scenario": scenario.name,"family":scenario.kind,"intensity":scenario.intensity,
            "seed":seed,"samples":len(frame),
            "mean_absolute_prediction_shift_ppm":float(np.nanmean(np.abs(values-base))),
            "ood_rate":diagnostics["out_of_domain_rows"]/max(1,len(frame)),
            "missing_columns":len(diagnostics["missing_raw_input_columns"])
        }
        if "Li_icpms" in frame:
            truth = pd.to_numeric(frame["Li_icpms"],errors="coerce").to_numpy()
            mask = np.isfinite(truth) & np.isfinite(values)
            if mask.any():
                diff = truth[mask]-values[mask]
                rec["mae_ppm"] = float(np.mean(np.abs(diff)))
                rec["rmse_ppm"] = float(np.sqrt(np.mean(diff**2)))
        records.append(rec)
    output = destination/"scenario_metrics.csv"
    pd.DataFrame(records).to_csv(output,index=False)
    manifest = {
        "schema_version":1,"seed":seed,"model":"model_1","scenario_count":len(records),
        "dataset_sha256":hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        "limitations":"Synthetic perturbations are not independent ground truth; extreme values may be physically impossible."
    }
    (destination/"scenario_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return output
