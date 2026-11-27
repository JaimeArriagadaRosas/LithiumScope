import os
from pathlib import Path
import sys
import time

from scripts.pytest_console import Progress, monitor, report_details, print_summary


def test_progress_counts_once_and_distinguishes_setup_error():
    lines = []
    progress = Progress(0, output=lambda line, **kwargs: lines.append(line))
    progress.consume({"kind": "collected", "total": 2}, 0)
    for node, phase, outcome in (("tests/cli/a.py::test_one", "call", "passed"),
                                  ("tests/cli/a.py::test_two", "setup", "failed")):
        progress.consume({"kind": "start", "nodeid": node}, 1)
        progress.consume({"kind": "report", "nodeid": node, "phase": phase, "outcome": outcome}, 2)
        progress.consume({"kind": "finish", "nodeid": node}, 3)
    assert progress.done == 2
    assert progress.counts["passed"] == 1 and progress.counts["error"] == 1
    assert any("2/2" in line for line in lines)
    assert any("[ERROR]" in line for line in lines)


def test_heartbeat_identifies_slow_test():
    lines = []
    p = Progress(0, output=lambda line, **kwargs: lines.append(line))
    p.consume({"kind":"collected","total":5}, 0)
    p.consume({"kind":"start","nodeid":"tests/core/a.py::test_slow"}, 1)
    p.heartbeat(10)
    assert "test_slow" in lines[-1] and "0/5" in lines[-1]


def test_summary_reads_actual_evidence(tmp_path,capsys):
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuite><testcase classname="tests.core.a" name="test_a" time="1"/>'
                     '<testcase classname="tests.core.a" name="test_b" time="2"><failure message="bad"/></testcase></testsuite>')
    coverage = tmp_path / "coverage.xml"
    coverage.write_text('<coverage line-rate="0.5708"/>')
    details = report_details(junit, coverage)
    assert details["line_coverage_pct"] == 57.08
    assert details["groups"]["tests.core"] == {"passed": 1, "failure": 1}
    assert details["slowest"][0]["seconds"] == 2
    print_summary({"counts":{"tests":2,"failures":1,"errors":0,"skipped":0},
        "details":details,"state":"failed","duration_seconds":3,"coverage_state":"enabled",
        "exit_code":1,"log":"pytest.log"}, tmp_path)
    output = capsys.readouterr().out
    assert "FALLIDO" in output and "57.08%" in output and "test_b" in output


def test_observer_subprocess_preserves_junit_and_log(tmp_path,capsys):
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    fixture = tmp_path / "test_sample.py"
    fixture.write_text("def test_ok():\n    assert True\n\ndef test_fail():\n    assert False\n")
    log, progress, junit = (tmp_path / name for name in ("pytest.log","progress.jsonl","junit.xml"))
    env = dict(os.environ, LITHIUMSCOPE_PYTEST_PROGRESS=str(progress), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    code = monitor([sys.executable,"-m","pytest","-p","scripts.pytest_progress",
                    str(fixture),"-q","--junitxml",str(junit)],
                   cwd=root,env=env,log=log,progress_file=progress)
    assert code == 1
    assert "assert False" in log.read_text(encoding="utf-8")
    assert '"kind": "finish"' in progress.read_text(encoding="utf-8")
    assert "2/2" in capsys.readouterr().out
    assert report_details(junit,tmp_path/"absent.xml")["nonpassing"][0]["state"]=="failure"


def test_live_line_rewrites_clears_and_never_wraps():
    from io import StringIO
    from scripts.pytest_console import LiveLine
    stream = StringIO()
    live = LiveLine(stream, columns=lambda: 20)
    live.update("12345678901234567890123456789", flush=True)
    live.update("OK", flush=True)
    live.clear()
    assert stream.getvalue() == "\r1234567890123456789\rOK" + " "*17 + "\r  \r"
    assert "\n" not in stream.getvalue()
    assert not live.active
    live.clear()
    assert stream.getvalue().endswith("\r  \r")
