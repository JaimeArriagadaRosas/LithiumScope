"""Parse real JUnit outcomes and canonical identity; never invent missing evidence."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .aggregation import aggregate
from .architecture import CATEGORIES, SCOPES


def _canonical_identity(case, known):
    props = {p.get("name"): p.get("value", "") for p in case.findall("./properties/property")}
    explicit = props.get("lithiumscope_test_path")
    if explicit is not None:
        parts = explicit.split("/")
        if (len(parts) == 4 and parts[0] == "tests" and parts[1] in CATEGORIES
                and ".." not in parts and parts[3].startswith("test_") and parts[3].endswith(".py")
                and (known is None or explicit in known)
                and props.get("lithiumscope_test_category") == parts[1]):
            marks = set(props.get("lithiumscope_test_markers", "").split(","))
            if marks & set(CATEGORIES) == {parts[1]} and marks <= set(CATEGORIES + SCOPES):
                return explicit, sorted(marks), True
        # Invalid explicit identities cannot be repaired by guessing a basename.
        return "", [], False
    if known is not None:
        module = case.get("classname") or case.get("name", "")
        normalized = module.replace(chr(92), "/").removesuffix(".py").replace("/", ".")
        candidates = [p for p in known if normalized in {
            p.removesuffix(".py").replace("/", "."),
            p.removeprefix("tests/").removesuffix(".py").replace("/", "."),
            Path(p).stem}]
        if len(candidates) == 1:
            return candidates[0], [], False
    return "", [], False


def report_details(junit, coverage, *, root=None):
    result = {"line_coverage_pct": None, "groups": {}, "nonpassing": [], "slowest": [],
              "collection_errors": 0, "executed_tests": 0, "collection_causes": {},
              "junit_state": "MISSING", "cases": [], **aggregate([])}
    known = None
    if root is not None:
        known = {p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py")}
    try:
        xml = ET.parse(junit).getroot()
        groups, timings, cases = {}, [], []
        for case in xml.iter("testcase"):
            module, case_name = case.get("classname", ""), case.get("name", "")
            path, markers, explicit = _canonical_identity(case, known)
            name = f"{path or module}::{case_name}" if path or module else case_name
            group = ("tests." + path.split("/")[1]) if path else (
                ".".join((module or case_name.replace("/", ".").replace(chr(92), ".")).split(".")[:2])
                or "unknown")
            error = case.find("error")
            collection = error is not None and error.get("message", "").lower() == "collection failure"
            if collection:
                result["collection_errors"] += 1
                raw = error.text or ""
                match = re.search(r"No module named ['\"]([^'\"]+)['\"]", raw)
                cause = f"ModuleNotFoundError: {match.group(1)}" if match else next(
                    (line.strip()[:500] for line in reversed(raw.splitlines()) if line.strip()), "collection failure")
                causes = result["collection_causes"]
                causes[cause] = causes.get(cause, 0) + 1
            else:
                result["executed_tests"] += 1
            status = next((tag for tag in ("error", "failure", "skipped") if case.find(tag) is not None), "passed")
            record = {"module": module, "test": case_name, "outcome": status,
                      "path": path, "markers": markers, "canonical_identity": explicit,
                      "phase": "collection" if collection else "execution"}
            cases.append(record)
            if status != "passed":
                child = case.find(status)
                message = child.get("message") or (child.text or "").strip().split("\n")[-1]
                result["nonpassing"].append({"test": name, "state": status,
                    "message": (cause if collection else message)[:500], "phase": record["phase"]})
            groups.setdefault(group, Counter())[status] += 1
            if not collection:
                timings.append({"test": name, "seconds": float(case.get("time", "0"))})
        result.update(aggregate(cases))
        result.update(junit_state="OK" if cases else "EMPTY", cases=cases,
                      groups={k: dict(v) for k, v in sorted(groups.items())},
                      slowest=sorted(timings, key=lambda x: x["seconds"], reverse=True)[:5])
    except (OSError, ET.ParseError, ValueError) as exc:
        result["junit_state"] = "MISSING" if isinstance(exc, FileNotFoundError) else "ERROR"
    try:
        result["line_coverage_pct"] = round(float(ET.parse(coverage).getroot().attrib["line-rate"]) * 100, 2)
    except (OSError, ET.ParseError, ValueError, KeyError):
        pass
    return result
