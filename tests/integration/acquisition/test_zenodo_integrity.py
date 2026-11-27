import hashlib
import json
from dataclasses import replace
import pytest

from lithiumscope.core.exceptions import DatasetError
from lithiumscope.datasets.management.registry import get_dataset_spec
from lithiumscope.tools.acquisition import tabular

def _entry(name, data):
    return {"key": name, "size": len(data), "checksum": "md5:" + hashlib.md5(data, usedforsecurity=False).hexdigest(),
            "links": {"content": "https://example.invalid/" + name}}

def test_zenodo_reuses_only_verified_complete_selection(tmp_path, monkeypatch):
    spec = replace(get_dataset_spec("greenpeg"), zenodo_files=("a", "b"))
    data = {"a": b"first", "b": b"second"}
    monkeypatch.setattr(tabular, "_zenodo_manifest", lambda _: [_entry(k,v) for k,v in data.items()])
    calls = []
    def download(url, target):
        key = url.rsplit("/",1)[-1]
        calls.append(key)
        target.write_bytes(data[key])
    monkeypatch.setattr(tabular, "_stream_download", download)
    (tmp_path / "a").write_bytes(data["a"])
    (tmp_path / "b").write_bytes(b"broken")
    tabular._zenodo_download(spec, tmp_path)
    assert calls == ["b"]
    assert json.loads((tmp_path/"zenodo_manifest.json").read_text())["record"] == spec.zenodo_record
    monkeypatch.setattr(tabular, "_zenodo_manifest", lambda _: pytest.fail("verified cache should work offline"))
    tabular._zenodo_download(spec, tmp_path)
    assert calls == ["b"]

def test_zenodo_incomplete_selection_and_corruption_fail(tmp_path, monkeypatch):
    spec = replace(get_dataset_spec("greenpeg"), zenodo_files=("a","b"))
    monkeypatch.setattr(tabular, "_zenodo_manifest", lambda _: [_entry("a",b"a")])
    with pytest.raises(DatasetError, match="selection incomplete"):
        tabular._zenodo_download(spec,tmp_path)
    spec = replace(spec, zenodo_files=("a",))
    monkeypatch.setattr(tabular, "_stream_download", lambda url,path: path.write_bytes(b"b"))
    with pytest.raises(DatasetError, match="checksum"):
        tabular._zenodo_download(spec,tmp_path)
    assert not (tmp_path/"zenodo_manifest.json").exists()
    assert not (tmp_path/"a").exists()
