"""Find retired import paths without importing or training any model."""
from __future__ import annotations

import ast
import json
from pathlib import Path


def _exists(src: Path, name: str) -> bool:
    target = src.joinpath(*name.split("."))
    return target.with_suffix(".py").is_file() or target.is_dir()


def find_obsolete_imports(root: Path) -> list[str]:
    root = Path(root)
    src = root / "src"
    mapping = json.loads((root / "docs/integral-refactor/module-relocation.json").read_text(encoding="utf-8"))
    retired = {old: new for old, new in mapping.items()
               if old.startswith(("lithiumscope.datasets.", "lithiumscope.experiments.analysis."))
               and not _exists(src, old)}
    findings = set()

    def examine(path, node, name):
        for old, new in retired.items():
            if name == old or name.startswith(old + "."):
                findings.add(f"{path.relative_to(root).as_posix()}:{node.lineno}: {name} -> {new}{name[len(old):]}")
                return
        if name == "lithiumscope.experiments.analysis" or name.startswith("lithiumscope.experiments.analysis."):
            findings.add(f"{path.relative_to(root).as_posix()}:{node.lineno}: retired analysis package: {name}")

    for folder in (src, root / "scripts", root / "tests"):
        for path in sorted(folder.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            except SyntaxError as exc:
                findings.add(f"{path.relative_to(root).as_posix()}:{exc.lineno}: syntax error: {exc.msg}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        examine(path, node, alias.name)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    # Prefer concrete imported submodule when it appears in the relocation catalog.
                    for alias in node.names:
                        qualified = node.module + "." + alias.name
                        if qualified in retired:
                            examine(path, node, qualified)
                        else:
                            examine(path, node, node.module)
                elif isinstance(node, ast.Call) and node.args:
                    func = node.func
                    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                    if name in {"import_module", "run_module", "__import__"}:
                        first = node.args[0]
                        if isinstance(first, ast.Constant) and isinstance(first.value, str):
                            examine(path, node, first.value)
    return sorted(findings)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = find_obsolete_imports(root)
    for finding in findings:
        print(finding)
    print(f"Retired internal imports or syntax errors: {len(findings)}")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
