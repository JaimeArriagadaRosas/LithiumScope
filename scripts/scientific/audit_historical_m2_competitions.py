"""Read-only comparison of archived M2 competitions sharing a verified cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verified_labels(data, folds):
    labels = np.full(len(data), -1)
    assignments = np.full(len(data), -1)
    for fold in folds:
        test = np.asarray(fold["test_indices"])
        train = np.asarray(fold["train_indices"])
        require(np.array_equal(np.sort(np.concatenate([train, test])), np.arange(len(data))),
                "Not a complete train/test partition")
        require(not (set(data.spatial_group.iloc[test]) & set(data.spatial_group.iloc[train])),
                "Overlapping spatial groups")
        require(np.all(assignments[test] == -1), "Duplicate test assignments")
        threshold = fold["label_threshold_ppm"]
        require(abs(np.quantile(data.Li_icpms.iloc[train], .75) - threshold) < 1e-10,
                "Archived threshold differs from TRAIN Q75")
        labels[test] = (data.Li_icpms.iloc[test] >= threshold).astype(int)
        assignments[test] = fold["fold"]
    require((assignments > 0).all(), "Incomplete test assignments")
    return labels, assignments


def compare(archive: Path, output: Path, reference: str, candidate: str) -> dict:
    archive = archive.resolve()
    output = output.resolve()
    require(not output.is_relative_to(archive), "Output must be outside the historical clone")
    manifest = archive / "data/processed/model_2/training_manifest.csv"
    data = pd.read_csv(manifest)
    digest = sha(manifest)
    require(len(data) == 706 and data.sample_id.is_unique, "Expected the frozen 706 cohort")
    sources = {str(manifest): digest}
    inventory, metrics, rf = [], [], {}
    for run in sorted((archive / "results/model_2/runs").iterdir()):
        if not run.is_dir():
            continue
        m = run / "manifests/dataset_manifest.json"
        f = run / "manifests/scientific_folds.json"
        rank = run / "tables/competition_ranking.csv"
        if not all(p.is_file() for p in (m, f, rank)):
            inventory.append({"run": run.name, "status": "missing_competition_evidence"})
            continue
        for p in (m, f, rank):
            sources[str(p)] = sha(p)
        meta = json.loads(m.read_text(encoding="utf-8-sig"))
        folds = json.loads(f.read_text(encoding="utf-8-sig"))
        ranking = pd.read_csv(rank)
        key = hashlib.sha256(json.dumps(folds, sort_keys=True).encode()).hexdigest()
        same_cohort = meta["source_sha256"] == digest
        inventory.append({
            "run": run.name, "rows": meta["rows"], "columns": meta["columns"],
            "dataset_sha256": meta["source_sha256"], "folds_sha256": key,
            "winner": str(ranking.iloc[0]["algorithm"]),
            "winner_mean_fold_auc": float(ranking.iloc[0]["roc_auc_mean"]),
            "status": "verified_706" if same_cohort else "separate_cohort",
        })
        if not same_cohort:
            continue
        y, fi = verified_labels(data, folds)
        for path in sorted((run / "tables").glob("oof_predictions_*.csv")):
            sources[str(path)] = sha(path)
            oof = pd.read_csv(path)
            scores = oof.prospectivity_score.to_numpy()
            algorithm = path.stem.removeprefix("oof_predictions_")
            require(len(oof) == len(data) and np.array_equal(oof.y_true, y)
                    and np.array_equal(oof.outer_fold, fi)
                    and np.allclose(oof.li_ppm, data.Li_icpms),
                    f"OOF identity mismatch: {path}")
            require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
                    f"Invalid probabilities: {path}")
            auc = np.mean([roc_auc_score(y[fi == k], scores[fi == k]) for k in np.unique(fi)])
            expected = ranking.loc[ranking.algorithm == algorithm, "roc_auc_mean"]
            if len(expected):
                require(abs(auc - float(expected.iloc[0])) < 1e-10, f"AUC mismatch: {path}")
            metrics.append({
                "run": run.name, "algorithm": algorithm, "mean_fold_auc": auc,
                "pooled_auc": roc_auc_score(y, scores), "pooled_ap": average_precision_score(y, scores),
                "pooled_brier": brier_score_loss(y, scores),
            })
            if algorithm == "random_forest":
                rf[run.name] = key, y, scores
    ka, y, sa = rf[reference]
    kb, yb, sb = rf[candidate]
    require(ka == kb and np.array_equal(y, yb), "Comparison folds or labels differ")
    groups = [np.flatnonzero(data.spatial_group.to_numpy() == group)
              for group in sorted(data.spatial_group.unique())]
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(2000):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[ix])) == 2:
            differences.append(roc_auc_score(y[ix], sb[ix]) - roc_auc_score(y[ix], sa[ix]))
    require(len(differences) > 0, "No valid paired group bootstrap replicates")
    require(all(sha(Path(p)) == v for p, v in sources.items()), "Historical input changed")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory).to_csv(output / "runs.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "verified_706_oof.csv", index=False)
    summary = {
        "runs": len(inventory),
        "complete_competitions": sum(x["status"] != "missing_competition_evidence" for x in inventory),
        "verified_706_runs": sum(x["status"] == "verified_706" for x in inventory),
        "verified_algorithm_oof": len(metrics),
        "comparison": {
            "reference": reference, "candidate": candidate,
            "pooled_auc_delta": float(roc_auc_score(y, sb) - roc_auc_score(y, sa)),
            "group_bootstrap_95_percentile_ci": np.quantile(differences, [.025, .975]).tolist(),
            "replicates": len(differences), "groups": len(groups),
            "scope": "retrospective paired evaluation; not controlled causal ablation; not independent",
        },
        "source_hashes": sources,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="training_20261113_133905_m0300")
    parser.add_argument("--candidate", default="training_20261113_200558_m0300")
    args = parser.parse_args()
    summary = compare(args.historical_root, args.output, args.reference, args.candidate)
    print(json.dumps({k: v for k, v in summary.items() if k != "source_hashes"}, indent=2))
