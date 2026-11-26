from hashlib import sha256
from zipfile import ZipFile
import pytest
from scripts.restore_scientific_evidence import restore


def test_restore_dry_run_and_no_overwrite(tmp_path):
    archive=tmp_path/"backup.zip"
    with ZipFile(archive,"w") as z:
        z.writestr("originales/a.csv",b"original")
    mapping={"a.csv":{"sha256":sha256(b"original").hexdigest(),"destination":"results/a.csv"}}
    path=tmp_path/"results/a.csv"
    assert restore(archive,tmp_path,mapping=mapping)[0]["state"]=="ready"
    assert not path.exists()
    assert restore(archive,tmp_path,mapping=mapping,apply=True)[0]["state"]=="restored"
    path.write_bytes(b"existing")
    assert restore(archive,tmp_path,mapping=mapping,apply=True)[0]["state"]=="conflict"
    assert path.read_bytes()==b"existing"
    mapping["a.csv"]["sha256"]="wrong"
    with pytest.raises(ValueError,match="checksum"):
        restore(archive,tmp_path,mapping=mapping,apply=True)
