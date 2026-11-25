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
