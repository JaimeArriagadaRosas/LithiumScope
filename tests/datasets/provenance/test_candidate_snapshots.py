import pandas as pd
import pytest

from lithiumscope.datasets.management.snapshots import materialize_sources, verify_snapshot
from lithiumscope.datasets.provenance.comparison import compare_versions
from lithiumscope.datasets.provenance.identity import observation_ids


def test_source_cache_rebuilds_only_changed_source(tmp_path):
    sources = {k: tmp_path / f"{k}.csv" for k in ("a", "b")}
    for path in sources.values():
        path.write_text("Li,Sample\n1,A\n")
    calls = []
    def transform(inputs, output):
        name, path = next(iter(inputs.items()))
        calls.append(name)
        output.write_bytes(path.read_bytes())
    def run():
        return materialize_sources(tmp_path / "cache", sources, policy={"version": 1},
                                   transform_revision="test-v1", transform=transform)
    first = run()
    assert calls == ["a", "b"]
    second = run()
    assert all(s.reused for s in second.values())
    sources["a"].write_text("Li,Sample\n2,A\n")
    third = run()
    assert calls == ["a", "b", "a"]
    assert third["a"].version_id != first["a"].version_id
    assert third["b"].version_id == first["b"].version_id
    third["a"].data_path.write_text("corruption")
    with pytest.raises(ValueError, match="integrity"):
        verify_snapshot(third["a"].directory)


def test_observation_identity_and_version_diff():
    before = pd.DataFrame({"source_row_position": [1, 2], "Li_icpms": [10., 20.]})
    ids = observation_ids(before, source="a", source_sha256="a"*64)
    assert ids.tolist() == observation_ids(before.iloc[::-1], source="a", source_sha256="a"*64).iloc[::-1].tolist()
    assert not ids.equals(observation_ids(before, source="b", source_sha256="a"*64))
    before["id"] = ids
    after = before.iloc[[1]].copy()
    after["Li_icpms"] = 25.
    result = compare_versions(before, after, identity=("id",))
    assert result["removed"] == 1 and result["modified"] == 1
    assert result["target_delta"]["max_abs"] == 5.
    with pytest.raises(ValueError, match="Duplicate"):
        compare_versions(before, pd.concat([after, after]), identity=("id",))
