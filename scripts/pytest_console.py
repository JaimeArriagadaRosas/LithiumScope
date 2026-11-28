"""Human-readable progress and results; pytest's complete output stays in its log."""
from __future__ import annotations

from collections import Counter
import json
import shutil
import sys
from pathlib import Path
import subprocess
import time


if __package__:
    from .test_reporting.outcomes import report_details
else:
    from test_reporting.outcomes import report_details


def elapsed(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def group_of(nodeid):
    path = nodeid.split("::", 1)[0].replace(chr(92), "/").split("/")
    if len(path) >= 4 and path[0] == "tests":
        return "/".join(path[1:3])
    return path[1] if len(path) >= 3 and path[0] == "tests" else path[0]


class LiveLine:
    """Replace one physical terminal line, including when the new text is shorter."""
    def __init__(self, stream, *, columns=None):
        self.stream = stream
        self.columns = columns or (lambda: shutil.get_terminal_size(fallback=(100, 24)).columns)
        self.active = False
        self.width = 0

    def update(self, line, **kwargs):
        # Reserve the final column so Windows CMD never wraps the progress line.
        limit = max(1, self.columns() - 1)
        line = line[:limit]
        padding = min(max(0, self.width - len(line)), max(0, limit - len(line)))
        self.stream.write("\r" + line + " " * padding)
        self.stream.flush()
        self.active = True
        self.width = len(line)

    def clear(self):
        if self.active:
            self.stream.write("\r" + " " * min(self.width, max(1, self.columns() - 1)) + "\r")
            self.stream.flush()
            self.active = False
            self.width = 0


class Progress:
    def __init__(self, start, *, output=print):
        self.start = start
        self.output = output
        self.total = None
        self.counts = Counter()
        self.outcomes = {}
        self.current = None
        self.group = None
        self.done = 0
        self.last_print = start
        self.last_done = 0
        self.phase = "Descubriendo pruebas y dependencias"

    def line(self, message, now):
        self.output(f"[{elapsed(now - self.start)}] {message}", flush=True)
        self.last_print = now
        self.last_done = self.done

    def status(self, now, *, heartbeat=False):
        if self.total is None:
            message = self.phase
        elif self.done == self.total:
            message = f"{self.done}/{self.total} | Finalizando informes y cobertura"
        else:
            pct = 100 * self.done / max(1, self.total)
            width = 20
            fill = min(width, int(pct * width / 100))
            message = (f"[{'#' * fill}{'-' * (width - fill)}] {pct:5.1f}% "
                       f"| {self.done}/{self.total} | OK {self.counts['passed']} "
                       f"| Fallos {self.counts['failure']} | Errores {self.counts['error']} "
                       f"| Omitidos {self.counts['skipped']} | {self.group or 'inicio'}")
            if heartbeat and self.current:
                message += " | En curso: " + self.current.split("::")[-1][:120]
        self.line(message, now)

    def consume(self, event, now):
        kind = event.get("kind")
        if kind == "collected":
            self.total = int(event["total"])
            self.line(f"Pruebas seleccionadas: {self.total} | módulos: {len(event.get('module_counts', {}))}", now)
        elif kind == "start":
            self.current = event["nodeid"]
            group = group_of(self.current)
            if group != self.group:
                self.group = group
                self.status(now)
        elif kind == "report":
            node = event["nodeid"]
            outcome = event["outcome"]
            if outcome == "failed":
                state = "failure" if event["phase"] == "call" else "error"
            elif outcome == "skipped":
                state = "skipped"
            else:
                state = "passed"
            priority = {"passed": 0, "skipped": 1, "failure": 2, "error": 3}
            previous = self.outcomes.get(node, "passed")
            self.outcomes[node] = max((state, previous), key=priority.get)
        elif kind == "finish":
            state = self.outcomes.pop(event["nodeid"], "unknown")
            self.done += 1
            self.counts[state] += 1
            self.current = None
            if state in {"failure", "error"}:
                self.line(f"[{'FALLO' if state == 'failure' else 'ERROR'}] {event['nodeid']}", now)
            if self.done == self.total or self.done - self.last_done >= max(1, (self.total or 100) // 20):
                self.status(now)
        elif kind == "session_finished":
            self.phase = "Cerrando proceso de validación"

    def heartbeat(self, now):
        if now - self.last_print >= 8:
            self.status(now, heartbeat=True)


def monitor(command, *, cwd, env, log, progress_file, show=True):
    """Poll a separate flushed event stream, avoiding pipe buffering/deadlocks."""
    live = LiveLine(sys.stdout) if show and sys.stdout.isatty() else None
    progress = Progress(time.monotonic(), output=live.update if live else print)
    if show:
        progress.line("Iniciando pytest; recopilando pruebas...", progress.start)
    progress_file.touch()
    with log.open("wb") as output, progress_file.open("r", encoding="utf-8") as events:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                   stderr=subprocess.STDOUT)
        try:
            while True:
                while True:
                    position = events.tell()
                    line = events.readline()
                    if not line or not line.endswith("\n"):
                        events.seek(position)
                        break
                    if show:
                        try:
                            progress.consume(json.loads(line), time.monotonic())
                        except (ValueError, KeyError, TypeError):
                            pass
                code = process.poll()
                if code is not None:
                    # Final events may have arrived between the last read and poll.
                    if show:
                        for line in events:
                            try:
                                progress.consume(json.loads(line), time.monotonic())
                            except (ValueError, KeyError, TypeError):
                                pass
                    return code
                if show:
                    progress.heartbeat(time.monotonic())
                time.sleep(.25)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            if live:
                live.clear()


def _print_groups(title, groups):
    if not groups:
        return
    print("\n" + title)
    print(f"{'Área':<28} {'OK':>7} {'Fallos':>7} {'Errores':>7} {'Omit.':>7}")
    for group, counts in groups.items():
        print(f"{group:<28} {counts.get('passed',0):7} {counts.get('failure',0):7} "
              f"{counts.get('error',0):7} {counts.get('skipped',0):7}")


def _interpret(summary, clean):
    state = summary["state"]
    counts, details = summary["counts"], summary["details"]
    if state == "passed":
        return "Suite aprobada; no demuestra superioridad científica."
    if state == "cancelled":
        return "Ejecución cancelada; conservar evidencia parcial."
    if state == "failed":
        return "Hay fallos o comprobaciones no superadas; revisar incidencias."
    if (state == "incomplete" and summary.get("pytest_exit_code") == 0
            and counts.get("tests",0) > 0 and counts.get("skipped",0) > 0
            and not counts.get("failures",0) and not counts.get("errors",0)
            and not details.get("collection_errors",0) and clean
            and details.get("collection",{}).get("full_suite_verified")
            and all(r["state"] == "OK" for r in summary.get("checks",{}).values())
            and (not summary.get("coverage_requested") or details.get("line_coverage_pct") is not None)
            and not summary.get("report_error")):
        return "Sin fallos; validación incompleta por pruebas omitidas."
    return "Evidencia incompleta; revisar recopilación, integridad, cobertura e incidencias."


def print_summary(summary, results):
    counts, details = summary["counts"], summary["details"]
    label = {"passed":"APROBADO","failed":"FALLIDO","cancelled":"CANCELADO",
             "incomplete":"INCOMPLETO"}.get(summary["state"],"DESCONOCIDO")
    clean = (summary.get("git_identity_unchanged") and summary.get("tracked_worktree_dirty") is False
             and summary.get("untracked_code_present") is False)
    checks = summary.get("checks",{})
    collection = details.get("collection",{})
    rate = details["line_coverage_pct"]
    print("\n" + "="*72 + f"\nVALIDACIÓN DEL SOFTWARE | {label}\n" + "="*72)
    print(f"Rama: {summary.get('git_branch') or 'N/D'} | Commit: {(summary.get('git_commit') or 'N/D')[:12]}")
    print(f"Duración: {elapsed(summary['duration_seconds'])} | Pytest: {summary.get('pytest_exit_code','N/D')} | Validación: {summary['exit_code']}")
    print("\n[1/5] RESULTADO GENERAL")
    if counts:
        passed = counts["tests"]-counts["failures"]-counts["errors"]-counts["skipped"]
        print(f"  Total: {counts['tests']} | OK: {max(0,passed)} | Fallos: {counts['failures']} | Errores: {counts['errors']} | Omitidos: {counts['skipped']}")
        print(f"  Casos ejecutados (no colección): {details.get('executed_tests',0)} | Errores de recopilación: {details.get('collection_errors',0)}")
    else:
        print("  JUnit no disponible; conteos no certificados.")
    print(f"  Cobertura de líneas: {rate:.2f}%" if rate is not None
          else f"  Cobertura: no disponible ({summary['coverage_state']})")
    print("  " + _interpret(summary,clean))
    print("\n[2/5] INTEGRIDAD Y COMPROBACIONES")
    print("  Git: " + ("commit limpio y sin cambios" if clean else "no certificado"))
    if collection:
        print(f"  Suite completa: {'verificada' if collection.get('full_suite_verified') else 'no verificada'}")
        print(f"  Módulos: {collection.get('collected_modules')}/{collection.get('expected_modules')} | Casos: {collection.get('selected_cases')} | Deseleccionados: {collection.get('deselected_cases')}")
        for module in collection.get("missing_modules",[]):
            print(f"  [FALTANTE] {module}")
        if collection.get("module_count_mismatches"):
            print(f"  Módulos con conteos diferentes: {len(collection['module_count_mismatches'])}")
    if checks:
        print(f"  Comprobaciones: {sum(r['state']=='OK' for r in checks.values())}/{len(checks)} OK")
        for name,record in checks.items():
            if record["state"] != "OK":
                print(f"  [{record['state']}] {name}: {record.get('log',record.get('scope',''))}")
    print("\n[3/5] COBERTURA DE PRUEBAS Y CRITERIOS")
    _print_groups("Categorías principales",details.get("categories") or details.get("groups",{}))
    _print_groups("Ámbitos transversales (solapados)",details.get("scopes",{}))
    if details.get("unclassified"):
        print(f"  Casos sin clasificación: {details['unclassified']}")
    criteria = details.get("criterion_summary",{})
    if criteria:
        print(f"\n  Evidencia de criterios: {criteria['state']}")
        print(f"  {'Familia':<10} {'Aprobados':>10} {'Incompletos':>12} {'Otros':>8}")
        for family,by_state in criteria.get("by_family",{}).items():
            ok=by_state.get("ASSOCIATED_TESTS_PASSED",0)
            incomplete=by_state.get("INCOMPLETE_TEST_EVIDENCE",0)
            other=sum(by_state.values())-ok-incomplete
            print(f"  {family:<10} {ok:>10} {incomplete:>12} {other:>8}")
            for status,n in by_state.items():
                if status not in {"ASSOCIATED_TESTS_PASSED","INCOMPLETE_TEST_EVIDENCE"}:
                    print(f"    {status}: {n}")
    print("\n[4/5] INCIDENCIAS")
    if summary.get("error"):
        print(f"  {summary['error']}")
    for cause,count in details.get("collection_causes",{}).items():
        print(f"  [RECOPILACIÓN] {count} | {cause}")
    failures = [r for r in details.get("nonpassing",[]) if r["state"]!="skipped"]
    skipped = [r for r in details.get("nonpassing",[]) if r["state"]=="skipped"]
    for row in failures:
        print(f"  [{row['state'].upper()}] {row['test']}\n    {' '.join(row['message'].split())}")
    if skipped:
        print(f"  Omitidas: {len(skipped)} (no equivalen a aprobadas)")
        for reason,n in Counter(" ".join(r["message"].split()) for r in skipped).items():
            print(f"    {n} | {reason or 'Motivo no disponible'}")
    if not failures and not skipped and not summary.get("error") and not details.get("collection_causes"):
        print("  Sin fallos, errores ni omisiones reportados." if counts else "  Incidencias no certificables sin JUnit.")
    if details.get("slowest"):
        slow=details["slowest"][0]
        print(f"  Prueba más lenta: {slow['seconds']:.2f} s | {slow['test'].split('::')[-1]}")
    print("\n[5/5] EVIDENCIAS")
    print(f"  Carpeta: {results}")
    names=["summary.json"]
    names.extend(name for name in ("junit.xml","coverage.xml","architecture.json","criterion_evidence.json",
                                   "verification_report.md") if (results/name).is_file())
    print("  Archivos: " + ", ".join(names))
    print(f"  Log pytest: {summary['log']}")
    print("  Detalle por responsabilidad, pruebas omitidas y tiempos: informes exportados.")
    print("="*72,flush=True)
