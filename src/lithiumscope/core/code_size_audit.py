"""Read-only physical-line audit for repository-owned modules.

python -m lithiumscope.core.code_size_audit [--strict] [--json PATH]
Do not enable strict CI until legacy responsibilities have been separated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ALLOWED = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
           ".ps1", ".bat", ".cmd", ".sh", ".sql", ".css", ".scss", ".html"}
SKIP = {"node_modules", ".venv", "venv", ".git", "__pycache__",
        "dist", "build", "results", "logs", "data", "models",
        ".cache", ".next", ".pytest_cache", ".ruff_cache"}
ROOTS = ("src", "tests")


def scan(root: Path, *, maximum: int = 300) -> list[dict]:
    findings = []
    for name in ROOTS:
        base = root / name
        if not base.exists():
            continue
        for folder, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP]
            for filename in files:
                path = Path(folder) / filename
                if path.suffix.lower() not in ALLOWED:
                    continue
                try:
                    lines = len(path.read_text(encoding="utf-8").splitlines())
                except UnicodeDecodeError:
                    continue
                if lines > maximum:
                    findings.append({
                        "path": path.relative_to(root).as_posix(),
                        "lines": lines, "excess": lines - maximum,
                    })
    return sorted(findings, key=lambda item: -item["lines"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-lines", type=int, default=300)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    findings = scan(root, maximum=args.max_lines)
    for item in findings:
        print(f"{item['lines']:4d}  {item['path']}")
    print(f"CODE_SIZE_AUDIT over_limit={len(findings)} max_lines={args.max_lines}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if args.strict and findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
