import json
from zipfile import ZipFile
import pandas as pd
import pytest
from lithiumscope.datasets.management.candidates import adapt_reference
from lithiumscope.datasets.management.snapshots import verify_snapshot
from lithiumscope.datasets.adapters.spectral.archives import list_text_members, selected_text_member

def test_reference_archive_member_is_versioned_without_extraction(tmp_path):
    archive = tmp_path / "spectra.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("field/sample.txt","400 .5\n500 .6\n")
        z.writestr("../unsafe.txt","unexpected")
    assert list_text_members(archive) == ["field/sample.txt"]
    with pytest.raises(ValueError, match="Unsafe"):
        with selected_text_member(archive, "../unsafe.txt"):
            pass
    opts = {"sample_id": "A", "instrument": "ASD FieldSpec 4"}
    result = adapt_reference(archive,key="fregeneda_almendra",options=opts,
                             root=tmp_path/"versions",member="field/sample.txt")
    assert verify_snapshot(result.directory)
    assert pd.read_csv(result.data_path).reflectance_fraction.tolist() == [.5,.6]
    evidence=json.loads((result.directory/"adapter_evidence.json").read_text())
    assert evidence["archive_member"] == "field/sample.txt"
    assert not evidence["training_label_certified"]
    assert adapt_reference(archive,key="fregeneda_almendra",options=opts,
                           root=tmp_path/"versions",member="field/sample.txt").reused
