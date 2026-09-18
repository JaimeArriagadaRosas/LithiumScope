from pathlib import Path


def test_repository_contains_no_notebooks():
    root = Path(__file__).resolve().parents[1]
    assert not list(root.rglob("*.ipynb"))


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
