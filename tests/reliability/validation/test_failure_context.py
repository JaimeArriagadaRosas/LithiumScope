import xml.etree.ElementTree as ET
from lithiumscope.tools.inspection.development.validation import _failure_context

def test_failure_context_retains_only_code_locations():
    element=ET.Element("failure",message="AssertionError: private data")
    element.text="C:\\private\\project\\tests\\test_a.py:15: AssertionError: sample xyz\nsrc/lithiumscope/core/paths.py:8: ValueError"
    report=_failure_context(element)
    assert report["locations"]==[{"file":"tests/test_a.py","line":15},{"file":"src/lithiumscope/core/paths.py","line":8}]
    assert "AssertionError" in report["exception_types"]
    assert "private" not in str(report) and "sample xyz" not in str(report)
