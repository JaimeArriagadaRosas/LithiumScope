"""Audit saved historical OOF records without fitting or changing models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(source: Path, destination: Path) -> dict:
    path = source / "sample_errors_six_algorithms.csv"
    data = pd.read_csv(path)
    if data[["version", "algorithm", "row"]].duplicated().any():
        raise ValueError("Duplicated OOF sample within algorithm and cohort")
    if not np.isfinite(data[["real", "pred", "train_q75"]]).all().all():
        raise ValueError("Nonfinite historical evidence")
    data["residual"] = data["pred"] - data["real"]
    data["squared_error"] = data.residual ** 2
    if not np.allclose(data.abs_error, data.residual.abs()):
        raise ValueError("Historical absolute errors are inconsistent")
    records, top = [], []
    counts = {"release_1120": 1120, "candidate_3420": 3420}
    for (version, algorithm), group in data.groupby(["version", "algorithm"]):
        expected = counts[version]
        if len(group) != expected or set(group.row) != set(range(expected)):
            raise ValueError(f"Incomplete OOF coverage: {version}/{algorithm}")
        high = group[group.real >= group.train_q75]
        ten = group.nlargest(10, "squared_error").copy()
        sse = group.squared_error.sum()
        ten["fraction_total_sse"] = ten.squared_error / sse if sse else np.nan
        top.append(ten)
        records.append({
            "version": version, "algorithm": algorithm, "n": len(group),
            "rmse": float(np.sqrt(group.squared_error.mean())),
            "mae": float(group.residual.abs().mean()), "bias": float(group.residual.mean()),
            "n_high": len(high), "high_bias": float(high.residual.mean()),
            "high_rmse": float(np.sqrt(high.squared_error.mean())),
            "top10_sse_pct": float(100 * ten.squared_error.sum() / sse) if sse else np.nan,
        })
    table = pd.DataFrame(records)
    if set(table.version) != set(counts) or not (table.groupby("version").algorithm.nunique() == 6).all():
        raise ValueError("Expected six algorithms in each historical cohort")
    ref = pd.read_csv(source / "regression_oof_six_algorithms.csv")
    compare = table.merge(ref, on=["version", "algorithm", "n"], suffixes=("", "_historical"),
                          validate="one_to_one", how="outer", indicator=True)
    if not (compare._merge == "both").all():
        raise ValueError("Historical metrics do not cover all evaluated algorithms")
    for metric in ("rmse", "mae", "bias"):
        if not np.allclose(compare[metric], compare[f"{metric}_historical"], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Historical {metric} mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / "summary.csv", index=False)
    pd.concat(top).to_csv(destination / "top_10_errors_per_algorithm.csv", index=False)
    meta = {
        "source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256((source / "regression_oof_six_algorithms.csv").read_bytes()).hexdigest(),
        "n_oof_records": len(data), "metrics_match_historical": True,
        "retrospective_only": True, "independent_holdout": False,
        "scope": "All six algorithms per historical cohort; no source chemistry join or causal attribution.",
    }
    (destination / "audit.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
