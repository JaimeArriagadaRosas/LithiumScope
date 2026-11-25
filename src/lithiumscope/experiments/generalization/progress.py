"""Console-only progress for diagnostic fits; no scientific decisions."""
from __future__ import annotations

from time import monotonic


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ScreenProgress:
    def __init__(self, group, algorithms, arms, outer_count, inner_count):
        self.total = len(algorithms) * outer_count * (len(arms) * inner_count + 2)
        self.done = self.ok = self.failed = self.skipped = 0
        self.started = monotonic()
        self.fit_started = self.started
        self.outer_count = outer_count
        self.inner_count = inner_count
        self.arm_count = len(arms)
        self.algorithm_count = len(algorithms)
        print(f"\n{'=' * 60}\nMODELO {group.removeprefix('model_')}\n{'=' * 60}")
        print(f"Algoritmos: {len(algorithms)} | Tratamientos: {len(arms)}")
        print(f"Folds exteriores: {outer_count} | Folds interiores: {inner_count}")
        print(f"Ajustes previstos (máximo): {self.total}\n", flush=True)

    def algorithm(self, index, name):
        print(f"\nALGORITMO {index}/{self.algorithm_count}: {name.upper()}", flush=True)

    def fold(self, fold):
        print(f"\n--- Fold exterior {fold}/{self.outer_count} ---", flush=True)

    def treatment(self, index, name):
        print(f"\nTratamiento {index}/{self.arm_count}: {name}", flush=True)

    def start(self, label):
        self.fit_started = monotonic()
        print(f"  [{self.done + 1}/{self.total}] {label} | En ejecución...", flush=True)

    def finish(self, *, result=None, metric=None, error=None):
        self.done += 1
        elapsed = duration(monotonic() - self.fit_started)
        if error is not None:
            self.failed += 1
            message = f"[ERROR] {error}"
        else:
            self.ok += 1
            message = f"[OK] {metric.upper()}={result[metric]:.4f} | TRAIN={result['train_'+metric]:.4f}"
        print(f"    {message} | {elapsed}", flush=True)
        self.status()

    def skip(self, count, reason):
        self.done += count
        self.skipped += count
        print(f"  [OMITIDO] {reason} ({count} ajustes no ejecutados)", flush=True)
        self.status()

    def status(self):
        percent = 100 * self.done / max(self.total, 1)
        print(f"    Progreso: {self.done}/{self.total} ({percent:.1f} %) | "
              f"Correctos: {self.ok} | Errores: {self.failed} | Omitidos: {self.skipped} | "
              f"Tiempo: {duration(monotonic()-self.started)}", flush=True)

    def selection(self, arm):
        print(f"\nTratamiento seleccionado por validación interna: {arm or 'N/D'}", flush=True)

    def end(self, state, destination):
        print(f"\n{'-' * 60}\nRESUMEN DEL MODELO")
        self.status()
        print(f"Estado experimental: {state}\nEvidencias: {destination}\n", flush=True)
