"""Associate observed pytest outcomes with the diagnostic's criterion identifiers."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


def expected_ids() -> set[str]:
    return {f"M1-{i:02}" for i in range(1, 28)} | {f"M2-{i:02}" for i in range(1, 31)} | {
        f"ARQ-{i:03}" for i in range(1, 110)
    }


def load_catalog(root: Path) -> dict:
    data = json.loads((root / "docs/integral-refactor/criterion-tests.json").read_text(encoding="utf-8"))
    criteria = data["criteria"]
    if data["schema_version"] != 1 or set(criteria) != expected_ids():
        raise ValueError("Incomplete or incompatible criterion catalog")
    for criterion, modules in criteria.items():
        if not 1 <= len(modules) <= 3 or len(set(modules)) != len(modules):
            raise ValueError(f"Invalid test associations: {criterion}")
        for relative in modules:
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve() / "tests") or not path.is_file():
                raise ValueError(f"Missing test module: {relative}")
    return criteria


def evaluate_catalog(criteria: dict, cases: list[dict], *, trusted_run: bool) -> dict:
    # Pytest's JUnit classname can contain package prefixes or test classes.
    by_module: dict[str, list[dict]] = {}
    for case in cases:
        parts = case["module"].split(".")
        for part in parts:
            if part.startswith("test_"):
                by_module.setdefault(part, []).append(case)
                break
    records = {}
    for criterion, modules in criteria.items():
        counts = Counter()
        missing = []
        for module in modules:
            observed = by_module.get(Path(module).stem, [])
            if not observed:
                missing.append(module)
            counts.update(case["outcome"] for case in observed)
        if not trusted_run:
            state = "UNVERIFIED_RUN"
        elif counts["failure"] or counts["error"]:
            state = "FAILED_TEST"
        elif missing or counts["skipped"] or not counts["passed"]:
            state = "INCOMPLETE_TEST_EVIDENCE"
        else:
            state = "ASSOCIATED_TESTS_PASSED"
        records[criterion] = {"state": state, "modules": modules, "counts": dict(counts),
                              "missing_modules": missing, "criterion_closed": False}
    return {"scope": "associated regression evidence only; scientific acceptance is separate",
            "summary": dict(Counter(r["state"] for r in records.values())), "criteria": records}
