"""Static, read-only audit of canonical test organization and its contracts."""
from __future__ import annotations

from collections import Counter
import ast
import json
from pathlib import Path
import tomllib
import shlex

CATEGORIES = ("architecture", "unit", "integration", "functional",
              "scientific", "regression", "performance", "reliability")
SCOPES = ("m1", "m2", "eda", "training", "slow", "local_artifacts")


def audit(root: Path) -> dict:
    root = root.resolve()
    tests = root / "tests"
    problems = []
    files = sorted(p for p in tests.rglob("*.py") if "__pycache__" not in p.parts)
    modules = {p.relative_to(root).as_posix() for p in files if p.name.startswith("test_")}
    categories = Counter()
    responsibilities = Counter()
    for module in sorted(modules):
        parts = module.split("/")
        if len(parts) != 4 or parts[1] not in CATEGORIES:
            problems.append(f"Noncanonical module: {module}")
            continue
        categories[parts[1]] += 1
        responsibilities["/".join(parts[1:3])] += 1
    for category in CATEGORIES:
        if not categories[category]:
            problems.append(f"Empty category: {category}")
    for name, count in Counter(Path(p).name for p in modules).items():
        if count != 1:
            problems.append(f"Duplicate basename: {name}")
    direct = Counter(p.parent for p in files)
    for folder, count in direct.items():
        if count > 20:
            problems.append(f"Directory limit: {folder.relative_to(root)} ({count})")
    for file in files:
        relative = file.relative_to(root).as_posix()
        try:
            source = file.read_text(encoding="utf-8")
            ast.parse(source, filename=relative)
            if len(source.splitlines()) > 300:
                problems.append(f"Line limit: {relative}")
        except (OSError, UnicodeError, SyntaxError) as exc:
            problems.append(f"Unreadable or invalid Python: {relative} ({type(exc).__name__})")
    if not (tests / "conftest.py").is_file():
        problems.append("Missing shared root conftest")
    if (tests / "fixtures/conftest.py").exists():
        problems.append("Obsolete nested fixture registration")
    original_count = reference_count = scope_count = 0
    try:
        docs = root / "docs/integral-refactor"
        relocation = json.loads((docs / "test-relocation-purpose.json").read_text(encoding="utf-8"))
        original_count = len(relocation)
        if len(set(relocation.values())) != original_count:
            problems.append("Duplicate canonical relocation destinations")
        for old, new in relocation.items():
            if new not in modules or (old != new and (root / old).exists()):
                problems.append(f"Invalid relocation: {old} -> {new}")
        catalog = json.loads((docs / "criterion-tests.json").read_text(encoding="utf-8"))
        expected = ({f"M1-{i:02}" for i in range(1, 28)}
                    | {f"M2-{i:02}" for i in range(1, 31)}
                    | {f"ARQ-{i:03}" for i in range(1, 110)})
        criteria = catalog["criteria"]
        if catalog["schema_version"] != 1 or set(criteria) != expected:
            problems.append("Invalid criterion catalog")
        for criterion, refs in criteria.items():
            reference_count += len(refs)
            if not refs or len(set(refs)) != len(refs) or not set(refs) <= modules:
                problems.append(f"Invalid criterion references: {criterion}")
        scopes = json.loads((docs / "test-marker-scopes.json").read_text(encoding="utf-8"))
        scope_count = len(scopes)
        for module, tags in scopes.items():
            if module not in modules or not set(tags) <= set(SCOPES) or len(set(tags)) != len(tags):
                problems.append(f"Invalid scope: {module}")
        with (root / "pyproject.toml").open("rb") as stream:
            config = tomllib.load(stream)["tool"]["pytest"]["ini_options"]
        registered = {m.split(":", 1)[0].strip() for m in config["markers"]}
        if not set(CATEGORIES + SCOPES) <= registered:
            problems.append("Missing registered markers")
        options = config.get("addopts", "")
        options = shlex.split(options) if isinstance(options, str) else options
        if config["testpaths"] != ["tests"] or any(
            token.startswith(("-m", "-k", "--ignore", "--deselect", "--lf", "--last-failed", "--collect-only"))
            for token in options
        ):
            problems.append("Default collection is not the full suite")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        problems.append(f"Catalog/configuration unavailable: {type(exc).__name__}: {exc}")
    return {"state": "ERROR" if problems else "OK", "scope": "static_test_architecture",
            "modules": len(modules), "original_modules": original_count,
            "category_modules": dict(sorted(categories.items())),
            "responsibility_modules": dict(sorted(responsibilities.items())),
            "max_direct_python_files": max(direct.values(), default=0),
            "criterion_references": reference_count, "scoped_modules": scope_count,
            "problems": problems}
