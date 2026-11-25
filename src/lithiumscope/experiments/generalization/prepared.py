"""Versioned, non-fitted data snapshots for paired development experiments."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import DATA_DIR
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.core.run_resume import build_training_signature


def identity(group, source, cfg):
    if group == "model_1":
        upstream = build_training_signature(group, source, ("app", group))
    elif group == "model_2":
        from lithiumscope.model_2.preparation.features.v4_inputs import build_scientific_training_signature
        upstream = build_scientific_training_signature(source, cfg)
    else:
        raise ValueError("Unknown model group")
    from importlib.metadata import version
    return canonical_json_hash({"upstream": upstream, "config": cfg,
                                "snapshot_code": file_sha256(Path(__file__)),
                                "versions": {p: version(p) for p in ("numpy","pandas","scikit-learn")}})


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _records(frame):
    return frame.astype(object).where(frame.notna(), None).values.tolist()


def audit_rows(x, y, groups, sources):
    table = x.copy()
    table["__target"] = y.to_numpy()
    table["__group"] = groups.to_numpy()
    table["__source"] = sources.to_numpy()
    signatures = pd.util.hash_pandas_object(x, index=False)
    target_ranges = y.groupby(signatures).transform("nunique")
    q1, q3 = y.quantile([.25, .75])
    median = y.median()
    mad = (y - median).abs().median()
    return pd.DataFrame({"row_id": x.index.astype(str), "group": groups.to_numpy(),
        "source": sources.to_numpy(), "target": y.to_numpy(),
        "missing_predictors": x.isna().sum(axis=1).to_numpy(),
        "duplicate_within_source_group": table.duplicated(keep=False).to_numpy(),
        "same_predictors_different_target": target_ranges.gt(1).to_numpy(),
        "target_iqr_extreme": ((y < q1 - 1.5*(q3-q1)) | (y > q3 + 1.5*(q3-q1))).to_numpy(),
        "target_mad_extreme": ((y-median).abs() > 3.5*mad/.67448975).to_numpy() if mad > 0 else False})


def save_snapshot(root, key, group, source, x, y, groups, sources, eligible, numeric, categorical, *, eda_path=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / key
    if destination.exists():
        load_snapshot(destination, key)
        return destination
    if (not x.index.is_unique or not x.index.equals(y.index) or not x.index.equals(groups.index)
            or not x.index.equals(sources.index) or groups.isna().any()
            or not np.isfinite(y.to_numpy(float)).all() or (y < 0).any()
            or np.asarray(eligible).dtype.kind != "b" or len(eligible) != len(x)
            or list(x.columns) != [*numeric, *categorical]):
        raise ValueError("Invalid prepared data identity")
    temporary = root / (".staging-" + uuid4().hex)
    temporary.mkdir()
    try:
        payload = {"columns": list(x.columns), "row_ids": x.index.astype(str).tolist(),
                   "values": _records(x), "target": y.to_numpy(float).tolist(),
                   "groups": groups.astype(str).tolist(), "sources": sources.astype(str).tolist(),
                   "eligible": np.asarray(eligible).tolist(), "numeric": numeric, "categorical": categorical}
        _write_json(temporary / "prepared.json", payload)
        audit = audit_rows(x, y, groups, sources)
        audit.to_csv(temporary / "row_audit.csv", index=False)
        summary = {"rows": len(x), "features": len(x.columns), "groups": int(groups.nunique()),
                   "eligible": int(np.sum(eligible)), "duplicates_flagged": int(audit.duplicate_within_source_group.sum()),
                   "conflicts_flagged": int(audit.same_predictors_different_target.sum()),
                   "iqr_extremes": int(audit.target_iqr_extreme.sum()), "mad_extremes": int(audit.target_mad_extreme.sum())}
        _write_json(temporary / "manifest.json", {"state": "complete", "identity": key, "model_group": group,
            "source_sha256": file_sha256(source), "eda_path": str(eda_path) if eda_path else None,
            "scope": "deterministic_preparation_only", "fitted_transform": False, "synthetic_rows": 0,
            "summary": summary, "artifacts": {p.name: file_sha256(p) for p in temporary.iterdir()}})
        temporary.rename(destination)
    except BaseException:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_snapshot(directory, expected):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (directory.is_symlink() or (directory/"manifest.json").is_symlink() or manifest.get("identity") != expected or manifest.get("state") != "complete"
            or manifest.get("scope") != "deterministic_preparation_only"
            or set(manifest.get("artifacts", {})) != {"prepared.json", "row_audit.csv"}):
        raise ValueError("Incompatible prepared snapshot")
    for name, digest in manifest["artifacts"].items():
        path = directory / name
        if path.is_symlink() or file_sha256(path) != digest:
            raise ValueError("Prepared snapshot integrity failure")
    data = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
    index = pd.Index(data["row_ids"])
    x = pd.DataFrame(data["values"], columns=data["columns"], index=index)
    for name in data["numeric"]:
        x[name] = pd.to_numeric(x[name], errors="raise").astype(float)
    for name in data["categorical"]:
        x[name] = x[name].astype(object).where(x[name].notna(), np.nan)
    return (x, pd.Series(data["target"], index=index), pd.Series(data["groups"], index=index),
            pd.Series(data["sources"], index=index), np.asarray(data["eligible"], dtype=bool),
            data["numeric"], data["categorical"])


def prepare_version(group, source, *, run_eda=True, root=None):
    """Use production preparation, never fit a data-dependent transform globally."""
    cfg = load_config(group)
    key = identity(group, source, cfg)
    root = Path(root) if root is not None else DATA_DIR / "processed" / group / "prepared_versions"
    destination = root / key
    if destination.exists():
        load_snapshot(destination, key)
        record = json.loads((destination/"manifest.json").read_text(encoding="utf-8"))
        if not run_eda:
            print(f"[OK] Versión preparada reutilizada: {destination}")
            return destination, None
        if record.get("eda_path"):
            from lithiumscope.results.eda_artifacts import verify_eda
            evidence = Path(record["eda_path"])
            try:
                eda_record = json.loads((evidence/"manifest.json").read_text(encoding="utf-8"))
                if verify_eda(evidence, eda_record["signature"]):
                    print(f"[OK] Versión preparada y EDA reutilizados: {destination}")
                    return destination, evidence
            except (OSError, ValueError, KeyError, TypeError):
                pass
    if group == "model_1":
        from lithiumscope.model_1.pipeline import prepare_training_data
        from lithiumscope.model_1.eda.pipeline import run_eda as eda
        from lithiumscope.core.scientific.eda_statistics import source_series
        prepared = prepare_training_data(source)
        x, y, groups = prepared.x, prepared.y, prepared.groups
        sources = source_series(prepared.frame)
        numeric, categorical = prepared.schema.numeric, prepared.schema.categorical
        eligible = np.ones(len(x), dtype=bool)
        evidence = eda(source, prepared=prepared, config=cfg) if run_eda else None
    else:
        from lithiumscope.model_2.preparation.features.v4_inputs import spectral_v4_enabled, load_spectral_v4_training_frame
        from lithiumscope.model_2.pipeline import load_model_2_training_frame
        from lithiumscope.model_2.eda.pipeline import run_eda as eda
        from lithiumscope.model_2.eda.quality import patch_quality
        from lithiumscope.core.scientific.eda_statistics import source_series
        frame = (load_spectral_v4_training_frame(source, cfg) if spectral_v4_enabled(cfg)
                 else load_model_2_training_frame(source)).reset_index(drop=True)
        manifest = pd.read_csv(source)
        target, group_col = cfg["training"]["lithium_column"], cfg["training"]["spatial_group_column"]
        y, groups = frame[target], frame[group_col]
        x = frame.drop(columns=[target, group_col])
        numeric, categorical = list(x.columns), []
        qa, _ = patch_quality(source, cfg, manifest)
        eligible = qa["train_eligible"].to_numpy(bool) if "train_eligible" in qa else np.ones(len(x), dtype=bool)
        sources = source_series(manifest)
        evidence = eda(source, frame=frame, config=cfg) if run_eda else None
        if "sample_id" in manifest:
            index = pd.Index(manifest["sample_id"].astype(str))
            x, y, groups, sources = [v.set_axis(index) for v in (x,y,groups,sources)]
    if identity(group, source, cfg) != key:
        raise ValueError("Prepared source changed during processing")
    destination = save_snapshot(root, key, group, source, x, y, groups, sources, eligible,
                                numeric, categorical, eda_path=evidence)
    summary = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))["summary"]
    print("\nVERSIÓN PREPARADA (sin ajuste ni aumentación global)")
    for name, value in summary.items():
        print(f"  {name}: {value}")
    print(f"  {destination}")
    return destination, evidence
