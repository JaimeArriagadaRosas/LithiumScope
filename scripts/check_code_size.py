"""Enforce 300-line modules, grandfathering explicit unfinished orchestrators.

Legacy exemptions are frozen and cannot grow. They remain an outstanding
refactoring obligation; an exemption is not evidence of good cohesion.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".ps1", ".bat", ".cmd", ".sql", ".mjs", ".cjs", ".css", ".scss", ".html", ".sh"}
def code_size_audit(
    root: Path,
    baseline: dict[str, int] | None = None,
    *,
    limit: int = 300,
) -> tuple[list[dict], list[str]]:
    """Audit the Git index: exactly the files that would be included in CI.

    Do not fail because of unrelated untracked work in the local workspace.
    New code is checked as soon as it is staged, and always in clean CI.
    """
    baseline = baseline or {}
    result = subprocess.run(
        ["git", "ls-files", "--cached"], cwd=root,
        capture_output=True, text=True, check=True,
    )
    inventory = []
    violations = []
    for relative in result.stdout.splitlines():
        if not relative.startswith(("src/", "tests/", "scripts/")):
            continue
        path = root / relative
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            continue
        try:
            length = len(path.read_text(encoding="utf-8").splitlines())
        except UnicodeDecodeError:
            continue
        inventory.append({"path": relative, "lines": length})
        allowed = max(limit, baseline.get(relative, limit))
        if length > allowed:
            violations.append(f"{relative}: {length} > {allowed}")
    return sorted(inventory, key=lambda row: row["lines"], reverse=True), violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    baseline_path = args.root / "config/code_size_legacy_exceptions.json"
    exceptions = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.is_file() else {}
    )
    inventory, errors = code_size_audit(args.root, exceptions)
    over = [item for item in inventory if item["lines"] > 300]
    print(f"Code files: {len(inventory)}; above 300: {len(over)}")
    for item in over:
        print(f"{item['lines']:4d}  {item['path']}")
    if errors:
        for error in errors:
            print("FAIL", error)
        if args.check:
            raise SystemExit(1)
    elif args.check:
        print("PASS: no new oversized files and no growth of old exceptions")


if __name__ == "__main__":
    main()
