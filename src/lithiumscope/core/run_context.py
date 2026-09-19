from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil

from lithiumscope.core.experiment_tracker import ExperimentTracker
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import CONFIG_DIR, RESULTS_DIR
from lithiumscope.core.run_resume import find_compatible_run


def _new_run_id() -> str:
    local = datetime.now().astimezone()
    offset = local.strftime("%z").replace("+", "p").replace("-", "m")
    return local.strftime("training_%Y%m%d_%H%M%S_") + offset


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
    checkpoints: Path
    tracker: ExperimentTracker
    resumed: bool = False

    @classmethod
    def create(
        cls,
        model_group: str,
        *,
        training_signature: str | None = None,
        resume: bool = True,
    ) -> "RunContext":
        existing = (
            find_compatible_run(model_group, training_signature)
            if resume and training_signature
            else None
        )
        if existing is not None:
            tracker = ExperimentTracker.load(existing)
            return cls(
                model_group=model_group,
                run_id=existing.name,
                root=existing,
                figures=existing / "figures",
                tables=existing / "tables",
                exports=existing / "exports",
                manifests=existing / "manifests",
                config_snapshot=existing / "config",
                checkpoints=existing / "checkpoints",
                tracker=tracker,
                resumed=True,
            )

        run_id = _new_run_id()
        root = RESULTS_DIR / model_group / "runs" / run_id
        figures = root / "figures"
        tables = root / "tables"
        exports = root / "exports"
        manifests = root / "manifests"
        config_snapshot = root / "config"
        checkpoints = root / "checkpoints"
        for directory in (
            root,
            figures,
            tables,
            exports,
            manifests,
            config_snapshot,
            checkpoints,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        config_hashes: dict[str, str] = {}
        for config_path in CONFIG_DIR.glob("*.yaml"):
            destination = config_snapshot / config_path.name
            shutil.copy2(config_path, destination)
            config_hashes[config_path.name] = file_sha256(destination)

        (config_snapshot / "config_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "files": config_hashes,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        tracker = ExperimentTracker(root, run_id, model_group)
        tracker.attach_summary(
            config_hashes=config_hashes,
            training_signature=training_signature,
            display_name=run_id,
        )
        return cls(
            model_group=model_group,
            run_id=run_id,
            root=root,
            figures=figures,
            tables=tables,
            exports=exports,
            manifests=manifests,
            config_snapshot=config_snapshot,
            checkpoints=checkpoints,
            tracker=tracker,
            resumed=False,
        )
