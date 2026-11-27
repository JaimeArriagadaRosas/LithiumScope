"""Static local import audit: distinguish eager, deferred and type-only edges."""
from __future__ import annotations

import ast
import json
from importlib.util import resolve_name
from pathlib import Path


def _module_name(root, path):
    parts = list(path.relative_to(root).with_suffix("").parts)
    # Dataset shims are imported via the owning package search path.
    if parts[:2] == ["lithiumscope", "datasets"] and "_legacy" in parts and parts[-1] != "__init__":
        parts.remove("_legacy")
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
    if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
        test=node.test
        if (isinstance(test.left,ast.Name) and test.left.id=="__name__"
                and len(test.ops)==len(test.comparators)==1 and isinstance(test.ops[0],ast.Eq)
                and isinstance(test.comparators[0],ast.Constant) and test.comparators[0].value=="__main__"):
            for child in node.body:
                yield from _imports(child,"entrypoint")
            for child in node.orelse:
                yield from _imports(child,scope)
            return
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
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


def _relocation_inventory(root, paths, unresolved):
    """Document real shims and retired paths without hiding unresolved references."""
    source=root/"docs/integral-refactor/module-relocation.json"
    if not source.is_file():
        return {"state":"not_available","retired":[],"existing_shims":[],"missing_destinations":[],"stale_references":[]}
    mapping=json.loads(source.read_text(encoding="utf-8"))
    retired,existing,missing,stale=[],[],[],[]
    for old,new in sorted(mapping.items()):
        seen={old}
        terminal=new
        while terminal in mapping:
            if terminal in seen:
                raise ValueError(f"Relocation mapping cycle: {old}")
            seen.add(terminal)
            terminal=mapping[terminal]
        item={"old":old,"destination":terminal}
        if terminal not in paths:
            missing.append(item)
        if old in paths:
            existing.append(item)
        else:
            retired.append(item)
            stale.extend({**edge,"canonical_destination":terminal} for edge in unresolved if edge["target"]==old)
    return {"state":"review_required" if missing or stale else "mapped",
            "retired":retired,"existing_shims":existing,"missing_destinations":missing,
            "stale_references":stale,
            "scope":"physical path inventory; compatibility and serialized model imports require runtime tests"}


def audit_imports(root: Path) -> dict:
    """Read ASTs only. Dynamic imports and conditional execution are not proven."""
    source = root / "src"
    paths = {_module_name(source, path): path for path in source.rglob("*.py")
             if "__pycache__" not in path.parts}
    graph = {name: set() for name in paths}
    relocation_file=root/"docs/integral-refactor/module-relocation.json"
    retired=set(json.loads(relocation_file.read_text(encoding="utf-8")))-set(paths) if relocation_file.is_file() else set()
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
        for node, scope in _imports(tree):
            if isinstance(node, ast.Call):
                if not node.args:
                    continue
                fn=node.func
                name=fn.id if isinstance(fn,ast.Name) else fn.attr if isinstance(fn,ast.Attribute) else ""
                arg=node.args[0]
                if name not in {"import_module","run_module"} or not isinstance(arg,ast.Constant) or not isinstance(arg.value,str) or not arg.value.startswith("lithiumscope."):
                    continue
                target=arg.value
                record={"source":module,"target":target,"scope":scope,"line":node.lineno}
                if target not in paths:
                    unresolved_local_imports.append(record)
                elif target!=module:
                    compatibility_edges.append(record)
                    edges.append(record)
                    if scope=="eager" and name=="import_module":
                        graph[module].add(target)
                continue
            # A from-import's base is a module, but the imported member can be a symbol.
            if isinstance(node,ast.Import):
                declared=[alias.name for alias in node.names]
            else:
                base=node.module or ""
                if node.level:
                    try:
                        base=resolve_name("."*node.level+base,package)
                    except ImportError:
                        base=""
                declared=[base]
                declared.extend(f"{base}.{alias.name}" for alias in node.names if f"{base}.{alias.name}" in retired)
            for target in declared:
                if target.startswith("lithiumscope.") and target not in paths:
                    unresolved_local_imports.append({"source":module,"target":target,"scope":scope,"line":node.lineno})
            for target in sorted(_targets(node,module,package,paths)):
                edges.append({"source":module,"target":target,"scope":scope,"line":node.lineno})
                if scope=="eager":
                    graph[module].add(target)
    external_references=[]
    script_catalog=root/"docs/integral-refactor/scientific-script-relocation.json"
    scripts=json.loads(script_catalog.read_text(encoding="utf-8")) if script_catalog.is_file() else {}
    # Scripts and tests are not runtime dependency-graph vertices, but references to
    # explicitly retired paths must be detected there as well.
    outside=[root/"main.py",*(root/"scripts").rglob("*.py"),*(root/"tests").rglob("*.py")]
    for path in sorted(set(outside)):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            tree=ast.parse(path.read_bytes())
        except (SyntaxError,OSError):
            continue  # the separate tracked syntax audit records these
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):
                names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:
                names=[node.module,*[f"{node.module}.{a.name}" for a in node.names]]
            elif isinstance(node,ast.Constant) and isinstance(node.value,str):
                names=[node.value.replace(chr(92),"/")]
            for name in names:
                if name in retired or name in scripts:
                    external_references.append({"path":path.relative_to(root).as_posix(),
                        "line":node.lineno,"target":name,"destination":scripts.get(name)})
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
            "unresolved_local_imports": unresolved_local_imports,
            "relocations": _relocation_inventory(root,paths,unresolved_local_imports),
            "retired_external_references": external_references,
            "deferred_edges": sum(e["scope"] == "deferred" for e in edges),
            "type_only_edges": sum(e["scope"] == "type_only" for e in edges),
            "scope": "static imports and literal import_module/run_module targets; computed imports and execution not proven"}
