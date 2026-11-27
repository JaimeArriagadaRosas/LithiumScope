from lithiumscope.tools.inspection.dependencies import audit_imports


def create(root, name, source):
    path = root / "src" / "lithiumscope" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_eager_cycles_are_reported_but_lazy_cycles_are_not(tmp_path):
    create(tmp_path, "a.py", "from lithiumscope import b\n")
    create(tmp_path, "b.py", "from lithiumscope import a\n")
    create(tmp_path, "c.py", "def lazy():\n    from lithiumscope import a\n")
    report = audit_imports(tmp_path)
    assert report["eager_cycles"] == [["lithiumscope.a", "lithiumscope.b"]]
    assert report["deferred_edges"] == 1


def test_independent_model_and_rendering_boundaries_are_audited(tmp_path):
    create(tmp_path, "model_1/a.py", "from lithiumscope.model_2 import b\n")
    create(tmp_path, "model_2/b.py", "pass\n")
    create(tmp_path, "reporting/a.py", "from lithiumscope.model_1 import a\n")
    create(tmp_path, "core/a.py", "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from lithiumscope.model_2 import b\n")
    report = audit_imports(tmp_path)
    assert len(report["cross_model_edges"]) == 1
    assert len(report["reporting_to_science_edges"]) == 1
    assert report["core_to_model_edges"] == []
    assert report["type_only_edges"] == 1


def test_relative_import_and_syntax_failure_are_visible(tmp_path):
    create(tmp_path, "a/__init__.py", "from . import x\n")
    create(tmp_path, "a/x.py", "from . import y\n")
    create(tmp_path, "a/y.py", "from . import x\n")
    create(tmp_path, "broken.py", "def invalid(")
    report = audit_imports(tmp_path)
    assert any({"lithiumscope.a.x", "lithiumscope.a.y"} <= set(c) for c in report["eager_cycles"])
    assert report["syntax_errors"][0]["type"] == "SyntaxError"


def test_literal_compatibility_imports_do_not_hide_cycles_or_missing_targets(tmp_path):
    create(tmp_path,"old.py",'from importlib import import_module\nimport_module("lithiumscope.new")\n')
    create(tmp_path,"new.py","from lithiumscope import old\n")
    create(tmp_path,"gone.py",'from importlib import import_module\nimport_module("lithiumscope.missing")\n')
    report=audit_imports(tmp_path)
    assert ["lithiumscope.new","lithiumscope.old"] in report["eager_cycles"]
    assert report["compatibility_edges"][0]["target"]=="lithiumscope.new"
    assert report["unresolved_literal_imports"][0]["target"]=="lithiumscope.missing"


def test_deferred_and_missing_static_imports_are_audited(tmp_path):
    create(tmp_path,"core/a.py",'def load():\n    from importlib import import_module\n    return import_module("lithiumscope.model_1.x")\n')
    create(tmp_path,"model_1/x.py",'import lithiumscope.core.a\nfrom lithiumscope.missing import symbol\n')
    report=audit_imports(tmp_path)
    assert report["eager_cycles"]==[]
    assert report["core_to_model_edges"][0]["scope"]=="deferred"
    assert report["unresolved_local_imports"][0]["target"]=="lithiumscope.missing"


def test_relocation_inventory_does_not_hide_retired_imports(tmp_path):
    import json
    create(tmp_path,"new.py","pass\n")
    create(tmp_path,"caller.py","from lithiumscope.old import symbol\n")
    catalog=tmp_path/"docs/integral-refactor/module-relocation.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({"lithiumscope.old":"lithiumscope.new"}),encoding="utf-8")
    report=audit_imports(tmp_path)
    assert report["unresolved_local_imports"][0]["target"]=="lithiumscope.old"
    assert report["relocations"]["stale_references"][0]["canonical_destination"]=="lithiumscope.new"
    assert report["relocations"]["state"]=="review_required"


def test_cli_guard_does_not_create_eager_import_cycle(tmp_path):
    create(tmp_path,"a.py",'if __name__=="__main__":\n    from lithiumscope import b\n')
    create(tmp_path,"b.py","from lithiumscope import a\n")
    assert audit_imports(tmp_path)["eager_cycles"]==[]


def test_retired_from_import_and_script_reference_detected(tmp_path):
    import json
    create(tmp_path,"core/__init__.py","")
    create(tmp_path,"core/new.py","pass")
    create(tmp_path,"caller.py","from lithiumscope.core import old\n")
    docs=tmp_path/"docs/integral-refactor"
    docs.mkdir(parents=True)
    (docs/"module-relocation.json").write_text(json.dumps({"lithiumscope.core.old":"lithiumscope.core.new"}))
    (docs/"scientific-script-relocation.json").write_text(json.dumps({"scripts/scientific/old.py":"scripts/scientific/audits/old.py"}))
    (tmp_path/"scripts").mkdir()
    (tmp_path/"scripts/check.py").write_text('path="scripts/scientific/old.py"\n')
    report=audit_imports(tmp_path)
    assert report["unresolved_local_imports"][0]["target"]=="lithiumscope.core.old"
    assert report["retired_external_references"][0]["target"]=="scripts/scientific/old.py"


def test_repository_refactor_has_no_import_boundary_blockers():
    from pathlib import Path
    report = audit_imports(Path(__file__).resolve().parents[2])
    for key in ("syntax_errors", "eager_cycles", "cross_model_edges",
                "core_to_model_edges", "reporting_to_science_edges",
                "unresolved_local_imports", "retired_external_references"):
        assert not report[key], (key, report[key])


def test_shared_helpers_have_canonical_owners():
    from lithiumscope.datasets.validation.tabular.missing_values import normalize_missing_values
    from lithiumscope.datasets.validation.tabular.schema import resolve_target
    from lithiumscope.core.scientific.json_reports import save_json_report
    assert callable(normalize_missing_values)
    assert callable(resolve_target)
    assert callable(save_json_report)


def test_historical_bootstrap_is_accessible_without_experiment_import():
    from lithiumscope.model_1.methodology.comparison_diagnostics import cluster_bootstrap_improvement
    report = cluster_bootstrap_improvement([1, 2], [2, 3], [1, 2], ["a", "b"], repeats=8)
    assert report["mae_gain"]["median"] == 1.0
    assert report["rmse_gain"]["median"] == 1.0
