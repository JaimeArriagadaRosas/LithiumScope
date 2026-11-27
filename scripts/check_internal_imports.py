"""Find retired import paths without importing or training any model."""
from __future__ import annotations

import ast
import json
from importlib.util import resolve_name
from pathlib import Path


def _exists(src: Path, name: str) -> bool:
    target = src.joinpath(*name.split("."))
    return target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()


def find_obsolete_imports(root: Path) -> list[str]:
    root = Path(root)
    src = root / "src"
    mapping = json.loads((root / "docs/integral-refactor/module-relocation.json").read_text(encoding="utf-8"))
    retired = {old: new for old, new in mapping.items()
               if old.startswith(("lithiumscope.datasets.", "lithiumscope.experiments.analysis.",
                                  "lithiumscope.experiments."))
               and not _exists(src, old)}
    retired_packages = tuple("lithiumscope.experiments." + name for name in (
        "acquisition", "georoc", "analysis", "model_1.targeted", "model_1.balanced",
        "model_1.augmentation", "model_1.expansion", "model_1.integral",
        "model_2.historical", "model_2.refinement_support"))
    findings = set()

    def examine(path, node, name):
        for old, new in retired.items():
            if name == old or name.startswith(old + "."):
                findings.add(f"{path.relative_to(root).as_posix()}:{node.lineno}: {name} -> {new}{name[len(old):]}")
                return
        if any(name == package or name.startswith(package + ".") for package in retired_packages):
            findings.add(f"{path.relative_to(root).as_posix()}:{node.lineno}: retired experimental package: {name}")
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
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        if not path.is_relative_to(src):
                            continue
                        parts = path.relative_to(src).with_suffix("").parts
                        package = ".".join(parts[:-1])
                        try:
                            base = resolve_name("." * node.level + base, package)
                        except ImportError:
                            continue
                    if not base:
                        continue
                    # Prefer concrete imported submodule when it appears in the relocation catalog.
                    for alias in node.names:
                        qualified = base + "." + alias.name
                        if qualified in retired or qualified in retired_packages:
                            examine(path, node, qualified)
                        else:
                            examine(path, node, base)
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
