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
        if not modules or len(set(modules)) != len(modules):
            raise ValueError(f"Invalid test associations: {criterion}")
        for relative in modules:
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve() / "tests") or not path.is_file():
                raise ValueError(f"Missing test module: {relative}")
    return criteria


def evaluate_catalog(criteria: dict, cases: list[dict], *, trusted_run: bool) -> dict:
    # Match canonical module paths first. Bare pytest module names are
    # accepted only when the stem is unique across the complete catalog.
    declared = sorted({module for modules in criteria.values() for module in modules})
    stems: dict[str, list[str]] = {}
    for module in declared:
        stems.setdefault(Path(module).stem, []).append(module)
    by_module: dict[str, list[dict]] = {}
    ambiguous = set()
    for case in cases:
        canonical = case.get("path")
        if canonical is not None:
            # Explicit identity is authoritative; never fall back to a misleading stem.
            if canonical in declared:
                by_module.setdefault(canonical, []).append(case)
            continue
        classname = case["module"]
        exact = []
        for module in declared:
            dotted = Path(module).with_suffix("").as_posix().replace("/", ".")
            # JUnit may omit the leading tests package name.
            names = {dotted, dotted.removeprefix("tests.")}
            if any(classname == name or classname.startswith(name + ".") for name in names):
                exact.append(module)
        if not exact:
            stem = next((part for part in classname.split(".") if part.startswith("test_")), "")
            exact = stems.get(stem, [])
        if len(exact) == 1:
            by_module.setdefault(exact[0], []).append(case)
        elif len(exact) > 1:
            ambiguous.update(exact)
    records = {}
    for criterion, modules in criteria.items():
        counts = Counter()
        missing = []
        for module in modules:
            observed = by_module.get(module, [])
            if not observed:
                missing.append(module)
            counts.update(case["outcome"] for case in observed)
        if not trusted_run:
            state = "UNVERIFIED_RUN"
        elif counts["failure"] or counts["error"]:
            state = "FAILED_TEST"
        elif missing or any(module in ambiguous for module in modules) or counts["skipped"] or not counts["passed"]:
            state = "INCOMPLETE_TEST_EVIDENCE"
        else:
            state = "ASSOCIATED_TESTS_PASSED"
        records[criterion] = {"state": state, "modules": modules, "counts": dict(counts),
                              "missing_modules": missing, "ambiguous_modules": sorted(set(modules) & ambiguous), "criterion_closed": False}
    return {"scope": "associated regression evidence only; scientific acceptance is separate",
            "summary": dict(Counter(r["state"] for r in records.values())), "criteria": records}
