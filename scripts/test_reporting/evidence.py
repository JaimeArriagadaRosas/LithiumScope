"""Persist traceable local test evidence and render a concise shareable report."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys


def collection_evidence(root, progress_file, cases, *, junit_state, require_full):
    selected = None
    deselected = 0
    finished = False
    try:
        with progress_file.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("kind") == "collected":
                    selected = event
                elif event.get("kind") == "deselected":
                    deselected += int(event["total"])
                elif event.get("kind") == "session_finished":
                    finished = True
    except OSError:
        pass
    inventory = {p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py")}
    module_counts = selected.get("module_counts", {}) if selected else {}
    observed = Counter(c["path"] for c in cases if c["phase"] == "execution" and c["canonical_identity"])
    execution_count = sum(c["phase"] == "execution" for c in cases)
    missing = sorted(inventory - set(module_counts))
    extra = sorted(set(module_counts) - inventory)
    mismatched = {p: {"collected": module_counts.get(p, 0), "reported": observed.get(p, 0)}
                  for p in sorted(set(module_counts) | set(observed))
                  if module_counts.get(p, 0) != observed.get(p, 0)}
    selected_count = selected.get("total") if selected else None
    complete = (selected is not None and finished and junit_state == "OK"
                and selected_count == execution_count == sum(observed.values())
                and not mismatched and not extra and deselected == 0 and not missing
                and not any(c["phase"] == "collection" for c in cases))
    return {"state": "OK" if complete else "INCOMPLETE",
            "scope": "full_suite" if require_full else "requested_selection",
            "full_suite_verified": complete, "expected_modules": len(inventory),
            "collected_modules": len(module_counts), "selected_cases": selected_count,
            "reported_execution_cases": execution_count, "deselected_cases": deselected,
            "session_finished": finished, "missing_modules": missing,
            "unexpected_modules": extra, "module_count_mismatches": mismatched,
            "canonical_execution_cases": sum(observed.values())}


def criterion_evidence(root, cases, *, trusted):
    # Bind to this checkout rather than an older installed package.
    source = str(root / "src")
    sys.path.insert(0, source)
    try:
        from lithiumscope.tools.inspection.development.criteria import evaluate_catalog, load_catalog
        criteria = load_catalog(root)
        records = [{**case, "path": case["path"]} for case in cases]
        result = evaluate_catalog(criteria, records, trusted_run=trusted)
        groups = {}
        for key, row in result["criteria"].items():
            groups.setdefault(key.split("-", 1)[0], Counter())[row["state"]] += 1
        result["by_family"] = {k: dict(v) for k, v in groups.items()}
        result["state"] = "OK" if trusted else "UNVERIFIED"
        result["trusted_run"] = trusted
        return result
    except (ImportError, OSError, ValueError, TypeError, KeyError) as exc:
        return {"state": "ERROR", "reason": type(exc).__name__, "trusted_run": False,
                "summary": {}, "by_family": {}, "criteria": {}}
    finally:
        sys.path.remove(source)


def write_json(path, data):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def _table(lines, title, groups):
    lines.extend(["", f"## {title}", "", "| Área | Aprobados | Fallos | Errores | Omitidos |",
                  "|---|---:|---:|---:|---:|"])
    for group, row in groups.items():
        values = [row.get(k, 0) for k in ("passed", "failure", "error", "skipped")]
        lines.append(f"| {group} | " + " | ".join(map(str, values)) + " |")


def write_report(path, summary, criteria):
    details = summary["details"]
    counts = summary["counts"]
    rate = details["line_coverage_pct"]
    lines = ["# Validación del software", "", f"- Estado: {summary['state']}",
             f"- Rama: {summary.get('git_branch')}", f"- Commit: {summary.get('git_commit')}",
             f"- Ejecución: {summary['run_id']}",
             f"- Duración: {summary['duration_seconds']} s",
             f"- Código pytest: {summary['pytest_exit_code']}",
             f"- Integridad Git: {summary['git_identity_unchanged'] and not summary['tracked_worktree_dirty'] and not summary['untracked_code_present']}",
             f"- Cobertura de líneas: {str(rate) + '%' if rate is not None else 'no disponible'}",
             "", "## Comprobaciones", ""]
    for name, record in summary["checks"].items():
        lines.append(f"- {name}: {record['state']}")
    collection = details.get("collection", {})
    lines += ["", "## Recopilación", "",
              f"- Suite completa verificada: {collection.get('full_suite_verified', False)}",
              f"- Módulos recopilados: {collection.get('collected_modules')}/{collection.get('expected_modules')}",
              f"- Casos seleccionados: {collection.get('selected_cases')}",
              f"- Conteos JUnit: {counts or 'no disponibles'}"]
    for module in collection.get("missing_modules", []):
        lines.append(f"- Módulo no recopilado: `{module}`")
    _table(lines, "Categorías principales", details["categories"])
    _table(lines, "Responsabilidades", details["responsibilities"])
    _table(lines, "Ámbitos transversales (solapados)", details["scopes"])
    if details["unclassified"]:
        lines += ["", f"Casos sin clasificación: {details['unclassified']}"]
    lines += ["", "## Criterios", "", f"- Evidencia: {criteria['state']}"]
    for family, states in criteria.get("by_family", {}).items():
        lines.append(f"- {family}: {states}")
    for key, row in criteria.get("criteria", {}).items():
        if row["state"] != "ASSOCIATED_TESTS_PASSED":
            lines.append(f"- {key}: {row['state']}; faltantes: {', '.join(row['missing_modules']) or 'ninguno'}")
    lines += ["", "## Incidencias", ""]
    for item in details["nonpassing"]:
        lines.append(f"- {item['state']} ({item['phase']}): `{item['test']}`")
    lines += ["", "## Evidencias", "", "- summary.json", "- junit.xml (si fue generado)",
              "- coverage.xml (si fue generado)", "- architecture.json (validación integral)",
              "- criterion_evidence.json", f"- Log: {summary['log']}", "",
              "Los tests asociados no acreditan por sí solos superioridad predictiva ni cierre científico."]
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
