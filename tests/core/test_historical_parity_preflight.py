from lithiumscope.tools.inspection.development.historical_parity import verify, _required

def test_missing_historical_inputs_are_explicitly_blocked(tmp_path):
    report=verify(tmp_path,tmp_path/"report.json")
    assert report["state"]=="BLOCKED"
    assert report["reason"]=="historical_artifacts_absent"
    assert len(report["missing"])==len(_required(tmp_path))
    assert not report["training"] and not report["activation"] and not report["acquisition"]
    assert (tmp_path/"report.json").is_file()
