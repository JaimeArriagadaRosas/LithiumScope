from pathlib import Path
import subprocess


EXPECTED_GITKEEP_PATHS = {
    "data/raw/model_1/.gitkeep",
    "data/raw/model_2/.gitkeep",
    "data/interim/model_1/.gitkeep",
    "data/interim/model_2/.gitkeep",
    "data/processed/model_1/.gitkeep",
    "data/processed/model_2/.gitkeep",
    "models/model_1/trained/.gitkeep",
    "models/model_1/metadata/.gitkeep",
    "models/model_2/trained/.gitkeep",
    "models/model_2/metadata/.gitkeep",
    "results/model_1/figures/.gitkeep",
    "results/model_1/metrics/.gitkeep",
    "results/model_1/predictions/.gitkeep",
    "results/model_2/figures/.gitkeep",
    "results/model_2/metrics/.gitkeep",
    "results/model_2/predictions/.gitkeep",
    "logs/.gitkeep",
}


def _tracked_paths(root: Path) -> set[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            path.relative_to(root).as_posix()
            for path in root.rglob(".gitkeep")
        }
    return {
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.strip().endswith("/.gitkeep")
    }


def test_repository_contains_no_notebooks():
    root = Path(__file__).resolve().parents[1]
    assert not list(root.rglob("*.ipynb"))


def test_repository_keeps_runtime_architecture_placeholders():
    root = Path(__file__).resolve().parents[1]
    actual = _tracked_paths(root)
    assert EXPECTED_GITKEEP_PATHS <= actual


def test_cli_menu_remains_thin():
    root = Path(__file__).resolve().parents[1]
    lines = (root / "src/lithiumscope/cli/menu.py").read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 80


def test_trainers_are_thin_entry_points():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "src/lithiumscope/model_1/training/trainer.py",
        root / "src/lithiumscope/model_2/training/trainer.py",
    ]
    assert all(len(path.read_text(encoding="utf-8").splitlines()) <= 40 for path in paths)


def test_runtime_lifecycle_is_separated():
    root = Path(__file__).resolve().parents[1]
    runtime = root / "src/lithiumscope/runtime"
    assert (runtime / "preboot.py").is_file()
    assert (runtime / "graceful_shutdown.py").is_file()
    assert (runtime / "lifecycle.py").is_file()
    assert (runtime / "console_status.py").is_file()
