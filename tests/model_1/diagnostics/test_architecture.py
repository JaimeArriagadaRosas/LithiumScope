"""Guard canonical M1 ownership and file-size limits."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
M1 = ROOT / "src/lithiumscope/model_1"


def test_m1_module_size_and_directory_limits():
    files = list(M1.rglob("*.py"))
    assert files
    oversized = [(str(p.relative_to(M1)), len(p.read_text(encoding="utf-8").splitlines()))
                 for p in files if len(p.read_text(encoding="utf-8").splitlines()) > 300]
    assert not oversized, oversized
    directories = {p.parent for p in files}
    crowded = [(str(p.relative_to(M1)), len(list(p.glob("*.py"))))
               for p in directories if len(list(p.glob("*.py"))) > 20]
    assert not crowded, crowded


def test_retired_m1_modules_are_absent():
    retired = (
        "steps", "training/tabnet.py", "training/trainer.py",
        "preflight.py", "preflight_run_model_1_preflight.py",
        "evaluation/domain.py", "evaluation/group_intervals.py",
        "evaluation/high_lithium.py", "evaluation/metrics.py",
        "evaluation/reports.py", "evaluation/robustness.py",
        "methodology/dual_release_metrics.py",
        "methodology/dataset_comparison_diagnostics.py",
        "prediction/input_parser.py",
    )
    assert not any((M1 / name).is_file() for name in retired)
    assert not list((M1 / "steps").rglob("*.py"))


def test_pipeline_has_no_preparation_reexports():
    tree = ast.parse((M1 / "pipeline.py").read_text(encoding="utf-8"))
    exported = {node.name for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert exported == {"run_training", "run_independent_evaluation"}
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom))
                   and any(alias.name in {"prepare_training_data", "prepare_prediction_data",
                                          "PreparedModel1Data"} for alias in node.names)
                   for node in tree.body)
