"""Separated validate_inputs responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.model_2.methodology.site_patch_lab import Path

def validate_inputs(root:Path):
    from lithiumscope.model_2.methodology.site_patch_lab import D_QC_REL, D_SUM_REL, HISTORICAL_N, Path, REF_DESIGN_REL, REF_REL, build_inputs, define_outer_fold, json, np, pd, sha
    contract, predictors, folds, _, source_hashes = build_inputs(root)
    qcpath=root/D_QC_REL
    qsumpath=root/D_SUM_REL
    reference=root/REF_REL
    design_path=root/REF_DESIGN_REL
    paths=(qcpath,qsumpath,reference,design_path)
    if not all(x.is_file() for x in paths):
        raise FileNotFoundError("Required frozen Phase C/D artifacts missing")
    qc=pd.read_csv(qcpath)
    summary=json.loads(qsumpath.read_text(encoding="utf-8"))
    design=json.loads(design_path.read_text(encoding="utf-8"))
    if not summary["sources_hashes_sha256"]["manifest"] == source_hashes["training_manifest.csv"]:
        raise RuntimeError("Physical QA and training manifest no longer match")
    if source_hashes["training_manifest.csv"] != design["source_sha256"]["training_manifest.csv"]:
        raise RuntimeError("Baseline and current training manifest differ")
    if source_hashes["spectral_features.csv"] != design["source_sha256"]["spectral_features.csv"]:
        raise RuntimeError("Baseline and current features differ")
    if source_hashes["scientific_folds.json"] != design["source_sha256"]["scientific_folds.json"]:
        raise RuntimeError("Baseline and frozen folds differ")
    if len(qc)!=783 or qc.sample_id.astype(str).tolist()!=contract.sample_id.astype(str).tolist():
        raise ValueError("QA inventory out of alignment with training manifest")
    if qc.manifest_index.to_numpy(int).tolist()!=list(range(len(qc))):
        raise ValueError("QA input order changed")
    if qc.image_sha256.isna().any() or not qc.image_available.all():
        raise ValueError("Missing original image in the QA inventory")
    if not qc.meta_sample_id_matches.all() or not qc.meta_scene_matches.all() or not qc.meta_coords_match.all():
        raise ValueError("Original image metadata mismatch")
    if not qc.meta_bands_match.all() or not qc.image_shape_ok.all():
        raise ValueError("Original spectral schema mismatch")
    if not int((qc.finite_pixel_fraction<.95).sum())==16:
        raise RuntimeError("Unexpected number of incomplete patches")
    # Compare current image contents to the audit hashes; no silent mutation
    # to the underlying original NPYs is allowed.
    for i,(p,h) in enumerate(zip(contract.image_path,qc.image_sha256)):
        if sha(Path(p))!=str(h):
            raise ValueError(f"Patch content changed since physical-label audit: {i}")
    baseline=pd.read_csv(reference)
    if (
        len(baseline)!=HISTORICAL_N or
        baseline.test_index.sort_values().tolist()!=list(range(HISTORICAL_N)) or
        baseline.arm.ne("legacy_all").any()
    ):
        raise ValueError("Incorrect archived baseline")
    # Explicit label and outer-fold identity checks independent of predictions:
    for i,fold in enumerate(folds,1):
        _,_,te,th=define_outer_fold(fold,contract)
        ref=baseline.set_index("test_index").loc[te]
        if ref.fold.ne(i).any():
            raise ValueError("Archived baseline outer-fold assignment changed")
        if not np.allclose(ref.fold_li_threshold_ppm.to_numpy(float),th,atol=1e-8):
            raise ValueError("Archived baseline Li threshold changed")
        if not np.array_equal(ref.y_true.to_numpy(int),(
                contract.Li_icpms.iloc[te].to_numpy(float)>=th).astype(int)):
            raise ValueError("Archived baseline labels changed")
    return contract,predictors,folds,qc,baseline,source_hashes,{
        "qc":sha(qcpath),"reference":sha(reference),"phase_c_design":sha(design_path),
    }
