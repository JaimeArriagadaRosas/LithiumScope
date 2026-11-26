"""Human-readable progress and results; pytest's complete output stays in its log."""
from __future__ import annotations

from collections import Counter
import json
import shutil
import sys
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


def elapsed(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def group_of(nodeid):
    path = nodeid.split("::", 1)[0].replace(chr(92), "/").split("/")
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
            self.line(f"Pruebas seleccionadas: {self.total}", now)
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


def report_details(junit, coverage):
    result = {"line_coverage_pct": None, "groups": {}, "nonpassing": [], "slowest": []}
    try:
        root = ET.parse(junit).getroot()
        groups = {}
        timings = []
        for case in root.iter("testcase"):
            name = case.get("classname", "") + "::" + case.get("name", "")
            group = ".".join(case.get("classname", "").split(".")[:2])
            counts = groups.setdefault(group, Counter())
            status = "passed"
            for tag in ("error", "failure", "skipped"):
                child = case.find(tag)
                if child is not None:
                    status = tag
                    message = child.get("message") or (child.text or "").strip().split("\n")[-1]
                    result["nonpassing"].append({"test": name, "state": status, "message": message[:500]})
                    break
            counts[status] += 1
            timings.append({"test": name, "seconds": float(case.get("time", "0"))})
        result["groups"] = {key: dict(value) for key, value in sorted(groups.items())}
        result["slowest"] = sorted(timings, key=lambda item: item["seconds"], reverse=True)[:5]
    except (OSError, ET.ParseError, ValueError):
        pass
    try:
        result["line_coverage_pct"] = round(float(ET.parse(coverage).getroot().attrib["line-rate"]) * 100, 2)
    except (OSError, ET.ParseError, ValueError, KeyError):
        pass
    return result


def print_summary(summary, results):
    counts = summary["counts"]
    details = summary["details"]
    label = {"passed": "APROBADO", "failed": "FALLIDO", "cancelled": "CANCELADO"}.get(summary["state"], "DESCONOCIDO")
    print("\n" + "=" * 76)
    print(f"VALIDACIÓN DEL SOFTWARE | {label}")
    print("=" * 76)
    print(f"Rama           : {summary.get('git_branch') or 'N/D'}")
    print(f"Commit         : {(summary.get('git_commit') or 'N/D')[:12]}")
    print(f"Duración       : {elapsed(summary['duration_seconds'])} (min:seg)")
    if counts:
        passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
        print(f"Total          : {counts['tests']}")
        print(f"Aprobados      : {max(0, passed)}")
        print(f"Fallidos       : {counts['failures']}")
        print(f"Errores        : {counts['errors']}")
        print(f"Omitidos       : {counts['skipped']}")
    else:
        print("Resultados     : JUnit no disponible; no se pueden certificar los conteos")
    rate = details["line_coverage_pct"]
    print(f"Cobertura      : {rate:.2f}% de líneas" if rate is not None
          else f"Cobertura      : no disponible ({summary['coverage_state']})")
    clean = (summary.get("git_identity_unchanged") and summary.get("tracked_worktree_dirty") is False
             and summary.get("untracked_code_present") is False)
    print("Integridad Git : " + ("commit limpio y sin cambios" if clean else "no certificada"))
    print(f"Código salida  : {summary['exit_code']}")
    if summary.get("error"):
        print(f"Incidencia     : {summary['error']}")
    if details["groups"]:
        print("\nRESULTADOS POR GRUPO")
        print(f"{'Grupo':32} {'OK':>7} {'Fallos':>7} {'Errores':>7} {'Omit.':>7}")
        for group, values in details["groups"].items():
            print(f"{group[:32]:32} {values.get('passed',0):7} {values.get('failure',0):7} "
                  f"{values.get('error',0):7} {values.get('skipped',0):7}")
    failures = [r for r in details["nonpassing"] if r["state"] != "skipped"]
    if failures:
        print("\nPRUEBAS CON FALLOS O ERRORES")
        for row in failures:
            print(f"- {row['test']}\n  {' '.join(row['message'].split())}")
    skipped = [r for r in details["nonpassing"] if r["state"] == "skipped"]
    if skipped:
        print("\nPRUEBAS OMITIDAS")
        for row in skipped:
            print(f"- {row['test']}\n  {' '.join(row['message'].split())}")
    if details["slowest"]:
        print("\nCINCO PRUEBAS MÁS LENTAS")
        for row in details["slowest"]:
            print(f"  {row['seconds']:7.2f} s | {row['test']}")
    print("\nEVIDENCIAS")
    print(f"  Resumen JSON : {results / 'summary.json'}")
    print(f"  JUnit XML    : {results / 'junit.xml'}")
    if rate is not None:
        print(f"  Cobertura XML: {results / 'coverage.xml'}")
    print(f"  Log completo : {summary['log']}")
    print("\nInterpretación : " + ("sin fallos de pytest; no demuestra superioridad científica."
          if summary["state"] == "passed" and counts else "revisar incidencias antes de continuar."))
    print("=" * 76, flush=True)
