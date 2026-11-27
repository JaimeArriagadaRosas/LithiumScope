"""Summarize disjoint primary categories and overlapping transversal scopes."""
from collections import Counter
from .architecture import CATEGORIES, SCOPES


def aggregate(cases):
    categories = {name: Counter() for name in CATEGORIES}
    scopes = {name: Counter() for name in SCOPES}
    responsibilities = {}
    unknown = Counter()
    for case in cases:
        status = case["outcome"]
        path = case.get("path", "")
        parts = path.split("/")
        if len(parts) != 4 or parts[0] != "tests" or parts[1] not in categories:
            unknown[status] += 1
            continue
        categories[parts[1]][status] += 1
        responsibilities.setdefault("/".join(parts[1:3]), Counter())[status] += 1
        for marker in set(case.get("markers", [])) & scopes.keys():
            scopes[marker][status] += 1
    def serialize(groups):
        return {key: dict(counts) for key, counts in groups.items()}
    return {"categories": serialize(categories),
            "responsibilities": serialize(dict(sorted(responsibilities.items()))),
            "scopes": serialize(scopes), "unclassified": dict(unknown),
            "scope_counts_overlap": True}
