"""Spectral stress experiments applied before the existing M2 feature extractor."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ImageScenario:
    name: str
    kind: str
    intensity: float
    seed: int = 42


def image_catalog(seed: int = 42) -> list[ImageScenario]:
    cases = [ImageScenario("baseline","baseline",0,seed)]
    cases.extend(ImageScenario(f"missing_pixels_{p:g}","missing_pixels",p,seed) for p in (.05,.10,.25,.50,.75))
    cases.extend(ImageScenario(f"noise_{p:g}","noise",p,seed) for p in (.01,.05,.10,.25))
    cases.extend(ImageScenario(f"band_{index}_missing","band_missing",index,seed) for index in range(6))
    cases.extend(ImageScenario(f"brightness_{p:g}","brightness",p,seed) for p in (.8,1.2,1.5))
    return cases


def perturb_image(image: np.ndarray, scenario: ImageScenario) -> np.ndarray:
    original = np.asarray(image,dtype=np.float32)
    if original.ndim != 3:
        raise ValueError("Expected image dimensions [bands,height,width]")
    out = original.copy()
    rng = np.random.default_rng(scenario.seed)
    if scenario.kind == "baseline":
        return out
    if scenario.kind == "missing_pixels":
        out[rng.random(out.shape) < scenario.intensity] = np.nan
    elif scenario.kind == "band_missing":
        index = int(scenario.intensity)
        if index >= len(out):
            raise ValueError("Band index out of range")
        out[index] = np.nan
    elif scenario.kind == "noise":
        scale = np.nanstd(out,axis=(1,2),keepdims=True)
        out += rng.normal(size=out.shape).astype(np.float32) * scale * scenario.intensity
    elif scenario.kind == "brightness":
        out *= scenario.intensity
    else:
        raise ValueError(f"Unknown image scenario: {scenario.kind}")
    return out


def evaluate_model_2_image(bundle: dict, image: np.ndarray, destination: Path, seed: int = 42) -> Path:
    """Reuse trained M2 estimator and feature schema, without fitting it."""
    from lithiumscope.model_2.steps.step_04_image_preprocessing import preprocess_image
    from lithiumscope.model_2.steps.step_05_spectral_features import extract_spectral_features,DEFAULT_BAND_NAMES
    from lithiumscope.model_2.schema import out_of_range_fraction
    destination.mkdir(parents=True,exist_ok=True)
    bands = bundle.get("band_names",DEFAULT_BAND_NAMES)
    expected = list(bundle["features"])
    rows = []
    for scenario in image_catalog(seed):
        disturbed = perturb_image(image,scenario)
        missing = float(np.mean(~np.isfinite(disturbed)))
        if np.any(np.all(~np.isfinite(disturbed),axis=(1,2))):
            rows.append({"scenario":scenario.name,"kind":scenario.kind,"intensity":scenario.intensity,
                "status":"ABSTAIN_ALL_BAND_NODATA","missing_pixel_fraction":missing})
            continue
        clean = preprocess_image(disturbed,normalize_per_band=bool(bundle.get("normalize_per_band",False)))
        features = extract_spectral_features(clean,band_names=bands)
        frame = pd.DataFrame([features])
        absent = [f for f in expected if f not in frame.columns]
        if absent:
            rows.append({"scenario":scenario.name,"kind":scenario.kind,"intensity":scenario.intensity,
                "status":"ABSTAIN_FEATURE_MISMATCH","missing_features":len(absent),"missing_pixel_fraction":missing})
            continue
        score = float(bundle["estimator"].predict_proba(frame[expected])[0,1])
        rows.append({"scenario":scenario.name,"kind":scenario.kind,"intensity":scenario.intensity,
            "status":"SCORED","missing_pixel_fraction":missing,"score":score,
            "ood_fraction":out_of_range_fraction(frame[expected],bundle.get("applicability_profile",{}))})
    output = destination/"image_scenario_metrics.csv"
    table = pd.DataFrame(rows)
    if "score" not in table:
        table["score"] = np.nan
    base = table.loc[table["scenario"]=="baseline","score"]
    if not base.empty:
        table["absolute_score_shift"] = (table["score"]-float(base.iloc[0])).abs()
    table.to_csv(output,index=False)
    return output
