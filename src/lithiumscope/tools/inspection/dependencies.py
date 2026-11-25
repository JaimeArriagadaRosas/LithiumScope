"""Static local import audit: distinguish eager, deferred and type-only edges."""
from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path


def _module_name(root, path):
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(node, scope="eager"):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        scope = "deferred"
    if isinstance(node, ast.If) and isinstance(node.test, (ast.Name, ast.Attribute)):
        name = node.test.id if isinstance(node.test, ast.Name) else node.test.attr
        if name == "TYPE_CHECKING":
            for child in node.body:
                yield from _imports(child, "type_only")
            for child in node.orelse:
                yield from _imports(child, scope)
            return
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        yield node, scope
    for child in ast.iter_child_nodes(node):
        yield from _imports(child, scope)


def _targets(node, module, package, available):
    if isinstance(node, ast.Import):
        names = [a.name for a in node.names]
    else:
        base = node.module or ""
        if node.level:
            try:
                base = resolve_name("." * node.level + base, package)
            except ImportError:
                return set()
        names = [base]
        names.extend(f"{base}.{a.name}" for a in node.names if a.name != "*")
    return {name for name in names if name in available and name != module}


def _strong_components(graph):
    """Tarjan SCC; each member list denotes a potential eager import cycle."""
    order, low, active, stack, components = {}, {}, set(), [], []
    def visit(node):
        order[node] = low[node] = len(order)
        stack.append(node)
        active.add(node)
        for target in sorted(graph[node]):
            if target not in order:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in active:
                low[node] = min(low[node], order[target])
        if low[node] == order[node]:
            component = []
            while stack:
                item = stack.pop()
                active.remove(item)
                component.append(item)
                if item == node:
                    break
            if len(component) > 1:
                components.append(sorted(component))
    for name in sorted(graph):
        if name not in order:
            visit(name)
    return sorted(components)


def audit_imports(root: Path) -> dict:
    """Read ASTs only. Dynamic imports and conditional execution are not proven."""
    source = root / "src"
    paths = {_module_name(source, path): path for path in source.rglob("*.py")
             if "__pycache__" not in path.parts}
    graph = {name: set() for name in paths}
    compatibility_edges = []
    unresolved_local_imports = []
    edges, syntax_errors = [], []
    for module, path in sorted(paths.items()):
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        try:
            tree = ast.parse(path.read_bytes(), filename=path.relative_to(root).as_posix())
        except (SyntaxError, OSError) as exc:
            syntax_errors.append({"path": path.relative_to(root).as_posix(), "type": type(exc).__name__})
            continue
        # Explicit compatibility shims use import_module with a literal target.
        # Include these as real dependency edges; do not conceal them from cycles.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else ""
            target = node.args[0]
            if name == "import_module" and isinstance(target, ast.Constant) and isinstance(target.value, str) and target.value.startswith("lithiumscope."):
                record = {"source": module, "target": target.value, "line": node.lineno}
                if target.value not in paths:
                    unresolved_local_imports.append(record)
                elif target.value != module:
                    compatibility_edges.append(record)
                    # Only literal shims in the top level or __main__ branch
                    # are treated as potential eager edges; conservative audit.
                    graph[module].add(target.value)
        for node, scope in _imports(tree):
            for target in sorted(_targets(node, module, package, paths)):
                edges.append({"source": module, "target": target, "scope": scope, "line": node.lineno})
                if scope == "eager":
                    graph[module].add(target)
    def under(name, layer):
        return name == f"lithiumscope.{layer}" or name.startswith(f"lithiumscope.{layer}.")
    cross_model = [e for e in edges if e["scope"] != "type_only" and (
        (under(e["source"], "model_1") and under(e["target"], "model_2")) or
        (under(e["source"], "model_2") and under(e["target"], "model_1")))]
    core_to_models = [e for e in edges if e["scope"] != "type_only" and under(e["source"], "core")
                      and any(under(e["target"], layer) for layer in ("model_1", "model_2", "prediction", "experiments"))]
    reporting_to_science = [e for e in edges if e["scope"] != "type_only" and under(e["source"], "reporting")
                            and any(under(e["target"], layer) for layer in ("model_1", "model_2", "prediction", "experiments"))]
    return {"modules": len(paths), "eager_cycles": _strong_components(graph),
            "cross_model_edges": cross_model, "core_to_model_edges": core_to_models,
            "reporting_to_science_edges": reporting_to_science, "syntax_errors": syntax_errors,
            "compatibility_edges": compatibility_edges,
            "unresolved_literal_imports": unresolved_local_imports,
            "deferred_edges": sum(e["scope"] == "deferred" for e in edges),
            "type_only_edges": sum(e["scope"] == "type_only" for e in edges),
            "scope": "static imports and literal import_module targets; computed imports and execution not proven"}
