"""Acquisition, overlap screening and separate lifecycle contracts."""
from pathlib import Path

import pandas as pd
import pytest

from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.management.sources.research import ResearchSource, ResearchPartition
from lithiumscope.tools.acquisition.georoc.research import acquire
from lithiumscope.experiments.data_lab.source_studies.georoc.candidate_eda import overlap_flags
from lithiumscope.experiments.data_lab import workflow


def test_registered_acquisition_reuses_bytes_and_rejects_changes(tmp_path):
    spec=ResearchSource("test","10.1/test",(ResearchPartition(123,"part.csv"),))
    calls=[]
    def download(identifier,path,*,expected_sha256=None):
        calls.append((identifier,path.exists()))
        if not path.exists():
            path.write_bytes(b"sample\n"*200)
        return {"sha256":file_sha256(path),"state":"cached" if calls[-1][1] else "downloaded"}
    files,_=acquire(spec,tmp_path,downloader=download)
    acquire(spec,tmp_path,downloader=download)
    assert calls==[(123,False),(123,True)]
    files[0].write_bytes(b"corruption")
    with pytest.raises(ValueError,match="changed"):
        acquire(spec,tmp_path,downloader=download)


def test_candidate_overlap_is_a_flag_not_training_selection():
    base=pd.DataFrame({"source_sample":["A"],"source_unique_id":["1"],"Longitude":[-70.],
                      "Latitude":[-30.],"Li_icpms":[700.]})
    new=pd.concat([base,base.assign(source_sample="B",source_unique_id="2",Li_icpms=1000.)],ignore_index=True)
    flags=overlap_flags(new,base)
    assert flags.coordinate_li_overlap.tolist()==[True,False]
    assert flags.source_unique_id_overlap.tolist()==[True,False]


def test_eda_does_not_prepare_treatments(monkeypatch,tmp_path):
    from lithiumscope.model_1.eda import pipeline
    evidence=tmp_path/"eda"
    evidence.mkdir()
    (evidence/"manifest.json").write_text('{"state":"completed"}',encoding="utf-8")
    from lithiumscope.results import eda_cache
    monkeypatch.setattr(eda_cache,"lookup",lambda *_:None)
    monkeypatch.setattr(eda_cache,"key_for",lambda *_:"test")
    monkeypatch.setattr(eda_cache,"record",lambda *_:None)
    monkeypatch.setattr(pipeline,"run_eda",lambda source:evidence)
    path,state,record=workflow.analyze("model_1",Path("source.csv"),include_candidates=False)
    assert path==evidence and state=="complete"
    assert not record["treatment_plans_created"] and record["model_fits"]==0


def test_cached_eda_skips_preparation(monkeypatch,tmp_path):
    from lithiumscope.results import eda_cache
    from lithiumscope.model_1.eda import pipeline
    (tmp_path/"manifest.json").write_text('{"state":"completed"}',encoding="utf-8")
    monkeypatch.setattr(eda_cache,"lookup",lambda *_:tmp_path)
    monkeypatch.setattr(pipeline,"run_eda",lambda *_:pytest.fail("No repeat EDA"))
    path,state,_=workflow.analyze("model_1",Path("source.csv"),include_candidates=False)
    assert path==tmp_path and state=="complete"


def test_source_registry_is_model_aware_without_automatic_m2_acquisition(monkeypatch):
    from lithiumscope.experiments.data_lab.source_studies import registry
    assert any(s.key=="m1_source_eda" and s.automatic_eda for s in registry.studies_for("model_1"))
    assert any(s.key=="georoc_candidate_eda" and not s.automatic_eda for s in registry.studies_for("model_1"))
    assert any(s.key=="georoc_external_evidence" for s in registry.studies_for("model_2"))
    monkeypatch.setattr(registry,"import_module",lambda *_:pytest.fail("Unexpected automatic M2 study"))
    assert registry.run_candidate_eda("model_2",Path("manifest.csv"))==[]


def test_eda_cache_rejects_corruption_and_wrong_owner(monkeypatch,tmp_path):
    from lithiumscope.results import eda_cache
    from lithiumscope.results.eda_artifacts import new_run,complete_run
    monkeypatch.setattr(eda_cache,"RESULTS_DIR",tmp_path)
    monkeypatch.setattr(eda_cache,"key_for",lambda *_:"source-key")
    source=tmp_path/"source.csv"
    source.write_text("Li\n10\n")
    run=new_run(tmp_path/"model_1"/"eda")
    (run/"tables"/"summary.csv").write_text("rows\n1\n")
    complete_run(run,signature_value="signature",details={
        "model_group":"model_1","source_sha256":file_sha256(source)})
    eda_cache.record("model_1",source,run,"source-key")
    assert eda_cache.lookup("model_1",source)==run
    with pytest.raises(ValueError,match="owner"):
        eda_cache.record("model_2",source,run,"source-key")
    (run/"tables"/"summary.csv").write_text("changed")
    assert eda_cache.lookup("model_1",source) is None


def test_missing_canonical_still_runs_candidate_sequence(monkeypatch):
    from lithiumscope.experiments.data_lab import workflow
    calls = []
    def candidates(group, source, **kwargs):
        calls.append((group, source, kwargs))
        return ([{"source": "chen", "state": "partial"}], [], None)
    monkeypatch.setattr(workflow, "run_candidate_eda", candidates)
    path, state, report = workflow.analyze("model_1", None, source_error="missing combined")
    assert path is None and state == "partial"
    assert report["canonical_state"] == "missing"
    assert calls == [("model_1", None, {"canonical_state": "missing", "detailed": True})]
    assert report["model_fits"] == 0


def test_local_georoc_inventory_continues_after_missing_partition(tmp_path):
    from lithiumscope.experiments.data_lab.source_studies.local_sources import georoc_files
    spec = ResearchSource("test", "10.1/test", (
        ResearchPartition(1, "first.csv"), ResearchPartition(2, "second.csv")))
    folder = tmp_path / "test"
    folder.mkdir()
    (folder / "second.csv").write_text("sample,li\nA,10\n")
    paths, inventory, receipt, planned = georoc_files(spec, tmp_path)
    assert paths == (folder / "second.csv",)
    assert [v["state"] for v in inventory] == ["missing", "unverified"]
    assert receipt is None and planned


def test_source_sequence_continues_and_excludes_retired_before_reading(monkeypatch, tmp_path):
    from lithiumscope.experiments.data_lab.source_studies import sequential, source_catalog, historical_evidence
    keys = source_catalog.known_keys()
    monkeypatch.setattr(source_catalog, "retirement",
        lambda key, **kw: {"reason": "closed", "decision_reference": "review"} if key == keys[1] else None)
    monkeypatch.setattr(historical_evidence, "inspect_history", lambda *args: [])
    calls = []
    def inspect(key, *args):
        calls.append(key)
        if key == keys[0]:
            raise FileNotFoundError("missing")
        return {"source": key, "state": "complete"}
    monkeypatch.setattr(sequential, "_inspect", inspect)
    records, retired, path = sequential.run_sources(None, results_root=tmp_path, canonical_state="missing")
    assert calls == [k for k in keys if k != keys[1]]
    assert records[0]["state"] == "blocked"
    assert retired[0]["source"] == keys[1]
    assert path.is_file()
    comparison = pd.read_csv(path.parent / "source_comparison.csv")
    assert comparison.source.tolist() == calls
    assert comparison.approved_rows.eq(0).all()


def test_source_retirement_requires_complete_evidence_and_can_reopen(tmp_path):
    from lithiumscope.experiments.data_lab.source_studies import source_catalog
    from lithiumscope.results.eda_artifacts import new_run, complete_run
    key = source_catalog.known_keys()[0]
    profile = new_run(tmp_path / "profiles")
    (profile / "tables" / "profile.csv").write_text("rows\n1\n")
    complete_run(profile, signature_value="test", details={"source_key": key,
        "all_originals_available": True, "source_sha256": {"part.csv": "digest"},
        "base_sha256": "base", "summary": {"rows": 1}})
    catalog = tmp_path / "catalog"
    with pytest.raises(ValueError, match="Explicit"):
        source_catalog.retire(key, profile, reason="no gain", decision_reference="review", root=catalog)
    source_catalog.retire(key, profile, reason="no gain", decision_reference="review",
                          all_uses_closed=True, root=catalog)
    record = source_catalog.retirement(key, root=catalog)
    assert record["profile"]["summary"] == {"rows": 1}
    archive = source_catalog.reopen(key, reason="new version", root=catalog)
    assert archive.is_file() and source_catalog.retirement(key, root=catalog) is None


def test_source_comparison_keeps_unknown_counts_and_qc_scopes_distinct():
    from lithiumscope.experiments.data_lab.source_studies.sequential import comparison_row
    blocked = comparison_row({"source": "georoc_andean_202609", "state": "blocked"})
    assert blocked["profile_rows"] is None and blocked["above_696_ppm"] is None
    row = comparison_row({"source": "chen_global_arcs_2020", "state": "complete",
        "profile": {"profile_rows": 10, "uncensored_numeric_positive_li": 10,
                    "basic_qc_rows": 8, "tail_above_40": 2, "tail_above_696": 0}})
    assert row["above_696_ppm"] == 0 and row["qc_definition"] == "source_specific_basic_qc"
    assert row["qc_no_identity_flag_rows"] is None and row["approved_rows"] == 0
