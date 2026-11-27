"""Classification and canonical identity for pytest, independent of import mode."""
from __future__ import annotations

from functools import lru_cache
import json
import re
from pathlib import Path

import pytest


CATEGORIES = frozenset({
    "architecture", "unit", "integration", "functional",
    "scientific", "regression", "performance", "reliability",
})
CROSS_MARKERS = frozenset({"m1", "m2", "eda", "training", "slow", "local_artifacts"})


@lru_cache(maxsize=None)
def load_scopes(root: Path) -> dict[str, list[str]]:
    path = root / "docs/integral-refactor/test-marker-scopes.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if any(not set(tags) <= CROSS_MARKERS for tags in data.values()):
        raise ValueError("Unknown transversal test marker")
    return data


def classify(root: Path, path: Path, scopes: dict[str, list[str]]) -> tuple[str, set[str]] | None:
    """Return canonical category and scope tags, or None for tests outside this suite."""
    try:
        parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return None
    if len(parts) < 2 or parts[0] != "tests":
        return None
    if len(parts) != 4 or parts[1] not in CATEGORIES:
        raise ValueError(f"Test outside canonical layout: {'/'.join(parts)}")
    relative = "/".join(parts)
    return relative, {parts[1], *scopes.get(relative, [])}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Assign markers before pytest applies its standard -m selection."""
    root = config.rootpath
    scopes = None
    for item in items:
        if not item.path.resolve().is_relative_to(root.resolve() / "tests"):
            continue
        if scopes is None:
            scopes = load_scopes(root)
        try:
            classification = classify(root, item.path, scopes)
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from exc
        if classification is None:
            continue
        relative, tags = classification
        # Named cases can specify narrower scopes inside mixed-purpose modules.
        name = getattr(item, "originalname", None) or item.nodeid.split("::")[-1].split("[")[0]
        for tag, pattern in (("m1", r"(?:^|_)(?:m1|model_1)(?:_|$)"),
                             ("m2", r"(?:^|_)(?:m2|model_2)(?:_|$)"),
                             ("eda", r"(?:^|_)eda(?:_|$)"),
                             ("training", r"(?:^|_)(?:training|train|pretraining|posttraining|pretrain|optuna|cv)(?:_|$)")):
            if re.search(pattern, name):
                tags.add(tag)
        existing = {mark.name for mark in item.iter_markers()}
        category = relative.split("/")[1]
        if (existing & CATEGORIES) - {category}:
            raise pytest.UsageError(f"Conflicting category: {item.nodeid}")
        for tag in sorted(tags - existing):
            item.add_marker(tag)
        markers = sorted(tags | (existing & CROSS_MARKERS))
        item.user_properties.extend([
            ("lithiumscope_test_path", relative),
            ("lithiumscope_test_category", category),
            ("lithiumscope_test_markers", ",".join(markers)),
        ])
