"""Read-only per-band and center/border missingness anatomy (M2 Phase E).

Only geometric validity masks are inspected. No image repair, interpolation,
new acquisition, imputation, model training or geologic certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.ablations import sha
from lithiumscope.model_2.methodology.site_patch_lab import D_QC_REL, D_SUM_REL


MANIFEST_REL="data/processed/model_2/vnext_expanded/training_manifest.csv"


def inspect_mask(arr:np.ndarray) -> dict:
    """Compute exact shared no-data footprint and central image coverage."""
    image=np.asarray(arr)
    if image.shape!=(6,32,32):
        raise ValueError("Expected 6-band 32x32 spectral input")
    valid=np.isfinite(image)
    center=valid[:,8:24,8:24]
    border_count=(32*32)-(16*16)
    border_fraction=(valid.sum()-center.sum())/(6*border_count)
    same_mask=np.all(valid==valid[:1],axis=0)
    invalid_pixel=(~valid).any(axis=0)
    y,x=np.where(invalid_pixel)
    out={
        "finite_fraction":float(valid.mean()),
        "finite_center_fraction":float(center.mean()),
        "finite_border_fraction":float(border_fraction),
        "all_six_bands_have_same_pixel_mask":bool(same_mask.all()),
        "fraction_pixels_sharing_validity_mask":float(same_mask.mean()),
        "fraction_pixels_invalid_in_any_band":float(invalid_pixel.mean()),
        "nonfinite_xmin":int(x.min()) if len(x) else None,
        "nonfinite_xmax":int(x.max()) if len(x) else None,
        "nonfinite_ymin":int(y.min()) if len(y) else None,
        "nonfinite_ymax":int(y.max()) if len(y) else None,
    }
    for i in range(6):
        out[f"band_{i}_finite"]=float(valid[i].mean())
    return out


def inspect_saved_patches(root:Path,out:Path)->dict:
    manifest=root/MANIFEST_REL
    qa_path=root/D_QC_REL
    qa_summary=root/D_SUM_REL
    if not (manifest.exists() and qa_path.exists() and qa_summary.exists()):
        raise FileNotFoundError("Frozen manifest or Phase D QA evidence missing")
    summary=json.loads(qa_summary.read_text(encoding="utf-8"))
    if sha(manifest)!=summary["sources_hashes_sha256"]["manifest"]:
        raise ValueError("Training manifest changed since physical QA")
    rows=pd.read_csv(manifest)
    qa=pd.read_csv(qa_path)
    if len(rows)!=783 or len(qa)!=783 or (
        qa.sample_id.astype(str).tolist()!=rows.sample_id.astype(str).tolist()
    ):
        raise ValueError("Manifest and Phase D QA order disagree")
    bad=qa.loc[qa.finite_pixel_fraction<.95]
    if len(bad)!=16:
        raise ValueError("Unexpected physical data QC sample count")
    findings=[]
    for item in bad.itertuples(index=False):
        index=int(item.manifest_index)
        raw_path=Path(str(rows.image_path.iloc[index]))
        if sha(raw_path)!=item.image_sha256:
            raise ValueError(f"Original image hash changed: {item.sample_id}")
        metrics=inspect_mask(np.load(raw_path,allow_pickle=False))
        if abs(metrics["finite_fraction"]-float(item.finite_pixel_fraction))>1e-12:
            raise ValueError("Per-patch mask result disagrees with Phase D QA")
        findings.append({"sample_id":item.sample_id,
                         "image_sha256":str(item.image_sha256),
                         **metrics})
    result=pd.DataFrame(findings)
    stats={
        "sample_count":len(result),
        "distinct_image_bytes":int(result.image_sha256.nunique()),
        "all_six_band_masks_equal_count":int(result.all_six_bands_have_same_pixel_mask.sum()),
        "center_below_95pct_count":int((result.finite_center_fraction<.95).sum()),
        "center_at_least_95pct_count":int((result.finite_center_fraction>=.95).sum()),
        "minimum_center_fraction":float(result.finite_center_fraction.min()),
        "maximum_center_fraction":float(result.finite_center_fraction.max()),
        "minimum_border_fraction":float(result.finite_border_fraction.min()),
        "maximum_border_fraction":float(result.finite_border_fraction.max()),
        "original_images_untouched":True,
        "interpreted_as_geologic_exposure":False,
        "manifest_sha256":sha(manifest),
    }
    out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/"missing_geometry_16.csv",index=False)
    (out/"missing_geometry_summary.json").write_text(
        json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    print("MISSING_GEOMETRY_COMPLETE",json.dumps(stats),flush=True)
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    out=Path(args.output)
    if not out.is_absolute():out=PROJECT_ROOT/out
    inspect_saved_patches(PROJECT_ROOT,out)


if __name__=="__main__":
    main()
