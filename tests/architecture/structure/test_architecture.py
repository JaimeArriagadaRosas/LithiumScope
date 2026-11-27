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
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    assert not list(root.rglob("*.ipynb"))


def test_repository_keeps_runtime_architecture_placeholders():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    actual = _tracked_paths(root)
    assert EXPECTED_GITKEEP_PATHS <= actual


def test_cli_menu_remains_thin():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    lines = (root / "src/lithiumscope/cli/menu.py").read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 80


def test_trainers_are_thin_entry_points():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    paths = [
        root / "src/lithiumscope/model_1/pipeline.py",
        root / "src/lithiumscope/model_2/pipeline.py",
    ]
    assert all(len(path.read_text(encoding="utf-8").splitlines()) <= 40 for path in paths)


def test_runtime_lifecycle_is_separated():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    runtime = root / "src/lithiumscope/runtime"
    assert (runtime / "preboot.py").is_file()
    assert (runtime / "graceful_shutdown.py").is_file()
    assert (runtime / "lifecycle.py").is_file()
    assert (runtime / "console_status.py").is_file()


def test_cli_contains_versioned_model_loader():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    menu = (
        root
        / "src/lithiumscope/cli/menu.py"
    ).read_text(encoding="utf-8")
    assert "4. Cargar modelo versionado" in menu
    assert "model_release_command.run()" in menu


def test_lab_modules_remain_focused():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    paths = [
        root / "src/lithiumscope/tools/inspection/pipeline_inspect.py",
        root / "src/lithiumscope/tools/inspection/pipeline_inspection_reporting.py",
        root / "src/lithiumscope/tools/inspection/pipeline_inspection_sources.py",
        root / "src/lithiumscope/tools/inspection/pipeline_inspection_models.py",
        root / "src/lithiumscope/tools/environment/lab_preboot.py",
        root / "src/lithiumscope/tools/environment/gpu_probe.py",
        root / "src/lithiumscope/tools/acquisition/georoc/filtered.py",
        root / "src/lithiumscope/datasets/adapters/georoc/ingestion.py",
        root / "src/lithiumscope/datasets/adapters/georoc/schema.py",
        root / "src/lithiumscope/datasets/adapters/georoc/transformation.py",
        root / "src/lithiumscope/tools/georoc/processing/georoc_raw_reader.py",
        root / "src/lithiumscope/tools/georoc/processing/georoc_normalizer.py",
        root / "src/lithiumscope/tools/georoc/processing/georoc_material_parser.py",
        root / "src/lithiumscope/tools/georoc/processing/georoc_material_filter.py",
        root / "src/lithiumscope/tools/georoc/processing/georoc_postprocess.py",
        root / "src/lithiumscope/tools/georoc/processing/georoc_source_checkpoint.py",
        root / "src/lithiumscope/tools/georoc/processing/georoc_schema_profile.py",
        root / "src/lithiumscope/tools/acquisition/georoc/query_contract.py",
        root / "src/lithiumscope/tools/acquisition/georoc/query_flow.py",
        root / "src/lithiumscope/tools/georoc/query/georoc_query_models.py",
        root / "src/lithiumscope/tools/georoc/query/georoc_query_actions.py",
        root / "src/lithiumscope/tools/georoc/query/georoc_query_diagnostics.py",
        root / "src/lithiumscope/tools/georoc/processing/georoc_dataset_health.py",
        root / "src/lithiumscope/tools/georoc/query/georoc_query_download.py",
        root / "src/lithiumscope/tools/georoc/processing/georoc_download_finalize.py",
        root / "src/lithiumscope/tools/georoc/query/georoc_query_download_links.py",
        root / "src/lithiumscope/tools/georoc/query/georoc_query_log.py",
        root / "src/lithiumscope/tools/georoc/query/georoc_query_transfer.py",
        root / "src/lithiumscope/tools/georoc/query/georoc_query_html.py",
        root / "src/lithiumscope/tools/georoc/query/georoc_query_payload.py",
        root / "src/lithiumscope/tools/georoc/query/georoc_query_export.py",
    ]
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) <= 300
        for path in paths
    )


def test_python_directories_are_focused():
    root=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    directories=[p for base in (root/"src/lithiumscope",root/"tests")
                 for p in (base,*base.rglob("*")) if p.is_dir() and "__pycache__" not in p.parts]
    oversized={str(p.relative_to(root)):len(list(p.glob("*.py"))) for p in directories
               if len(list(p.glob("*.py")))>20}
    assert not oversized,oversized


def test_core_and_renderer_do_not_depend_on_model_execution():
    from lithiumscope.tools.inspection.dependencies import audit_imports
    root=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    report=audit_imports(root)
    assert not report["syntax_errors"],report["syntax_errors"]
    assert not report["core_to_model_edges"],report["core_to_model_edges"]
    assert not report["reporting_to_science_edges"],report["reporting_to_science_edges"]


def test_canonical_relocation_destinations_exist():
    import json
    root=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    reloc=json.loads((root/"docs/integral-refactor/module-relocation.json").read_text(encoding="utf-8"))
    missing=[]
    for old,new in reloc.items():
        path=root/"src"/new.replace(".","/")
        if not path.with_suffix(".py").is_file() and not (path/"__init__.py").is_file():
            missing.append((old,new))
    assert not missing,missing


def test_testing_relocation_catalog_is_consistent():
    import json
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    catalog = json.loads((root / "docs/integral-refactor/test-relocation-purpose.json").read_text(encoding="utf-8"))
    categories = ["architecture","functional","integration","performance","regression","reliability","scientific","unit"]
    assert len(set(catalog.values())) == len(catalog)
    for old, new in catalog.items():
        destination = Path(new)
        assert destination.parts[0] == "tests"
        assert destination.parts[1] in categories
        assert len(destination.parts) == 4
        assert Path(old).name == destination.name
        assert (root / new).is_file(), (old, new)
        assert old == new or not (root / old).exists(), (old, new)
