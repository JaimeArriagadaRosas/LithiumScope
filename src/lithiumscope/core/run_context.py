from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil

from lithiumscope.core.paths import CONFIG_DIR, RESULTS_DIR
from lithiumscope.results.experiment_tracker import ExperimentTracker


@dataclass(frozen=True)
class RunContext:
    model_group: str
    run_id: str
    root: Path
    figures: Path
    tables: Path
    exports: Path
    manifests: Path
    config_snapshot: Path
    tracker: ExperimentTracker

    @classmethod
    def create(cls, model_group: str) -> "RunContext":
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = RESULTS_DIR / model_group / "runs" / run_id
        figures = root / "figures"
        tables = root / "tables"
        exports = root / "exports"
        manifests = root / "manifests"
        config_snapshot = root / "config"
        for directory in (
            root,
            figures,
            tables,
            exports,
            manifests,
            config_snapshot,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        for config_path in CONFIG_DIR.glob("*.yaml"):
            shutil.copy2(config_path, config_snapshot / config_path.name)

        tracker = ExperimentTracker(root, run_id, model_group)
        return cls(
            model_group,
            run_id,
            root,
            figures,
            tables,
            exports,
            manifests,
            config_snapshot,
            tracker,
        )
