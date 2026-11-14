"""Four-stage M1 diagnostic from actual archived releases and candidate runs.

No model activation, no Optuna. Frozen OOF metrics for six algorithms and
fresh refits only for train/test gap and grouped learning curves for the
SVM winner. The historical source must rebuild with exact original SHA256.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from lithiumscope.core.device import detect_device
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.datasets.georoc_ingestion import ingest_georoc_files
from lithiumscope.datasets.harmonization import merge_harmonized_sources
from lithiumscope.model_1.steps.step_01_load_data import load_data
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.model_1.methodology.dual_release_metrics import met, four

OUT = PROJECT_ROOT / "results/scientific_simulations/m1_dual_diagnostics_20261113"
VERSIONS = {
    "release_1120": (
        "training_20261105_204029_m0300",
        "2f99d38960c31a02b85aa5328c910c3aeab5ec1760cf6ef035922fbb56d41ea1",
    ),
    "candidate_3420": (
        "training_20261106_111653_m0300",
        "94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94",
    ),
}
ALGS = ("random_forest", "xgboost", "svm", "tabnet", "hist_gradient_boosting", "catboost")
FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def old_source(tmp):
    raw = PROJECT_ROOT / "data/raw/model_1/georoc/GEOROC_Andean_Arc_LithiumScope.csv"
    base = PROJECT_ROOT / "data/raw/model_1/Mamani09_Table_DR2.csv"
    georoc = ingest_georoc_files([raw], Path(tmp) / "georoc_original.csv", iron_total_mode="original")
    a = load_data(base, quiet=True)
    if "source_dataset" not in a:
        a["source_dataset"] = "Mamani09 bootstrap"
    b = load_data(georoc.path, quiet=True)
    m, _ = merge_harmonized_sources([a, b])
    dest = Path(tmp) / "training_combined_original.csv"
    m.to_csv(dest, index=False)
    return dest


def prepare(name, path):
    run, sha = VERSIONS[name]
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
        raise ValueError("Cannot reconstruct exact source SHA256 for " + name)
    data = prepare_training_data(path)
    expected = int(name.split("_")[1])
    if len(data.y) != expected or len(data.schema.numeric + data.schema.categorical) != 33:
        raise ValueError("Unexpected row/feature count")
    oof = pd.read_csv(PROJECT_ROOT / "results/model_1/runs" / run / "tables/oof_predictions_svm__identity.csv")
    if not np.array_equal(data.y.to_numpy(float), oof.Li_icpms_real.to_numpy(float)):
        raise ValueError("Labels/order not identical to saved competition")
    return data


def splits(name, data):
    run = VERSIONS[name][0]
    folder = PROJECT_ROOT / "results/model_1/runs" / run / "checkpoints/svm__identity"
    out = []
    seen = np.zeros(len(data.y), int)
    for j in range(1, 6):
        d = json.loads((folder / f"fold_{j:02d}.json").read_text("utf8"))
        te = np.asarray(d["test_indices"], int)
        tr = np.setdiff1d(np.arange(len(data.y)), te)
        if set(data.groups.iloc[tr]) & set(data.groups.iloc[te]):
            raise ValueError("Spatial overlap in held-out fold")
        seen[te] += 1
        out.append((j, tr, te, d["params"]))
    if not np.all(seen == 1):
        raise ValueError("Invalid OOF coverage")
    frozen = list(GroupKFold(5).split(data.x, data.y, data.groups))
    if any(not np.array_equal(te, frozen[j - 1][1]) for j, tr, te, params in out):
        raise ValueError("Frozen split reconstruction differs")
    return out


def analyze(name, data):
    run = VERSIONS[name][0]
    folder = PROJECT_ROOT / "results/model_1/runs" / run / "tables"
    y = data.y.to_numpy(float)
    s = splits(name, data)
    src = data.frame.get("source_dataset", pd.Series(["unknown"] * len(y))).fillna("unknown").astype(str).to_numpy()
    source_counts = pd.Series(src).value_counts().to_dict()
    print("DATA_READY", name, len(y), data.groups.nunique(), source_counts, flush=True)
    ids = np.zeros(len(y), int)
    cutoff = {}
    for j, tr, te, _ in s:
        ids[te] = j
        cutoff[j] = float(np.quantile(y[tr], 0.75))
    fold_rows = []
    all_rows = []
    sample_rows = []
    gap = []
    curves = []
    for alg in ALGS:
        df = pd.read_csv(folder / f"oof_predictions_{alg}__identity.csv")
        if len(df) != len(y) or not np.array_equal(df.Li_icpms_real.to_numpy(float), y):
            raise ValueError("Unaligned OOF for " + name + " " + alg)
        pred = df.Li_icpms_predicted.to_numpy(float)
        all_rows.append({"version": name, "algorithm": alg, **met(y, pred)})
        for j, tr, te, _ in s:
            c = cutoff[j]
            fold_rows.append(
                {
                    "version": name,
                    "algorithm": alg,
                    "fold": j,
                    "train_n": len(tr),
                    "test_n": len(te),
                    **four(y[te], pred[te], c),
                    **{f"reg_{k}": v for k, v in met(y[te], pred[te]).items()},
                }
            )
        for i in range(len(y)):
            c = cutoff[ids[i]]
            oc = "VP" if y[i] >= c and pred[i] >= c else "FN" if y[i] >= c else "FP" if pred[i] >= c else "VN"
            sample_rows.append(
                {
                    "version": name,
                    "algorithm": alg,
                    "row": i,
                    "fold": int(ids[i]),
                    "group": str(data.groups.iloc[i]),
                    "source": src[i],
                    "real": y[i],
                    "pred": pred[i],
                    "abs_error": abs(y[i] - pred[i]),
                    "train_q75": c,
                    "outcome": oc,
                }
            )
    device = detect_device(prefer_gpu=False)
    for j, tr, te, params in s:
        train_groups = data.groups.iloc[tr].to_numpy()
        unique = np.unique(train_groups)
        order = np.random.default_rng(20261113 + j).permutation(unique)
        oof_winner = pd.read_csv(folder / "oof_predictions_svm__identity.csv").Li_icpms_predicted.to_numpy(float)
        for frac in (0.25, 0.5, 0.75, 1.0):
            chosen = set(order[: max(2, int(np.ceil(len(unique) * frac)))])
            local = np.flatnonzero(np.isin(train_groups, list(chosen)))
            if len(chosen & set(data.groups.iloc[te])):
                raise ValueError("Learning curve leaked TEST")
            model = build_pipeline("svm", device, 42, data.schema, params, target_transform="identity")
            model.fit(data.x.iloc[tr[local]], y[tr[local]])
            ptr = model.predict(data.x.iloc[tr[local]])
            pte = model.predict(data.x.iloc[te])
            tm = met(y[tr[local]], ptr)
            em = met(y[te], pte)
            record = {
                "version": name,
                "fold": j,
                "fraction": frac,
                "train_rows": len(local),
                "train_groups": len(chosen),
                "test_rows": len(te),
                **{f"train_{k}": v for k, v in tm.items()},
                **{f"test_{k}": v for k, v in em.items()},
            }
            curves.append(record)
            if frac == 1:
                diff = float(np.max(abs(pte - oof_winner[te])))
                gap.append(
                    {
                        **record,
                        "train_test_rmse_gap": float(em["rmse"] - tm["rmse"]),
                        "train_test_mae_gap": float(em["mae"] - tm["mae"]),
                        "refit_vs_saved_oof_max_abs": diff,
                        "original_oof_reproduced": bool(diff < 1e-4),
                    }
                )
            print(
                "LEARNING",
                name,
                j,
                frac,
                "TRAIN_RMSE",
                round(tm["rmse"], 4),
                "TEST_RMSE",
                round(em["rmse"], 4),
                flush=True,
            )
    dist = {
        "version": name,
        "n": len(y),
        "groups": int(data.groups.nunique()),
        "median": float(np.median(y)),
        "q75": float(np.quantile(y, 0.75)),
        "q90": float(np.quantile(y, 0.9)),
        "q95": float(np.quantile(y, 0.95)),
        "q99": float(np.quantile(y, 0.99)),
        "max": float(np.max(y)),
        "std": float(np.std(y)),
        "over100": int((y > 100).sum()),
        "over500": int((y > 500).sum()),
        "over1000": int((y > 1000).sum()),
        "sources": source_counts,
    }
    return {
        "folds": fold_rows,
        "overall": all_rows,
        "samples": sample_rows,
        "gaps": gap,
        "curves": curves,
        "distribution": dist,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="m1_exact_release_") as t:
        release = analyze("release_1120", prepare("release_1120", old_source(t)))
        expanded = analyze(
            "candidate_3420", prepare("candidate_3420", PROJECT_ROOT / "data/processed/model_1/training_combined.csv")
        )
    gathered = {}
    for key in ("folds", "overall", "samples", "gaps", "curves"):
        gathered[key] = pd.DataFrame(release[key] + expanded[key])
    folds = gathered["folds"]
    cms = []
    for (v, alg), subset in folds.groupby(["version", "algorithm"]):
        counts = subset[["VP", "VN", "FP", "FN"]].sum().astype(int)
        tp, tn, fp, fn = (int(counts[k]) for k in ["VP", "VN", "FP", "FN"])

        def rate(a, b):
            return float(a / b) if b else None

        cms.append(
            {
                "version": v,
                "algorithm": alg,
                "VP": tp,
                "VN": tn,
                "FP": fp,
                "FN": fn,
                "precision": rate(tp, tp + fp),
                "recall": rate(tp, tp + fn),
                "specificity": rate(tn, tn + fp),
                "f1": rate(2 * tp, 2 * tp + fp + fn),
                "balanced_accuracy": (0.5 * (rate(tp, tp + fn) + rate(tn, tn + fp)) if tp + fn and tn + fp else None),
            }
        )
    by_source = []
    for (v, source), sub in gathered["samples"].query("algorithm=='svm'").groupby(["version", "source"]):
        by_source.append({"version": v, "source": source, **met(sub.real, sub.pred)})
    files = {
        "regression_oof_six_algorithms.csv": gathered["overall"],
        "confusion_fold_train_q75.csv": gathered["folds"],
        "confusion_six_algorithms.csv": pd.DataFrame(cms),
        "sample_errors_six_algorithms.csv": gathered["samples"],
        "train_test_gap_svm.csv": gathered["gaps"],
        "learning_curves_svm.csv": gathered["curves"],
        "regression_by_source_svm.csv": pd.DataFrame(by_source),
    }
    for fn, df in files.items():
        df.to_csv(OUT / fn, index=False)
        print("RESULT_FILE", fn, len(df), flush=True)
    meta = {
        "status": "completed_retrospective_scientific_diagnostic",
        "version_info": VERSIONS,
        "distribution": [release["distribution"], expanded["distribution"]],
        "training_source_sha_verified": True,
        "five_folds_spatial_oof_verified": True,
        "confusion_threshold": "train-only q75 each fold, not a primary classification objective",
        "learning_curves": "SVM winner only, 5 folds x 4 spatial-group fractions per version",
        "train_test_gaps": "SVM winner only, fold hyperparams frozen from original Optuna",
        "no_new_hyperparameter_search": True,
        "external_holdout": False,
        "no_active_model_changes": True,
    }
    (OUT / "audit_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE", str(OUT), flush=True)


if __name__ == "__main__":
    main()
