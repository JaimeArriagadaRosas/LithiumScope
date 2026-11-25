import hashlib
import json
import zipfile

import pytest
import requests

from lithiumscope.distribution import transport
from lithiumscope.distribution.local_archive import local_release


def test_retry_transient_tls_preserves_failure(monkeypatch):
    calls=[]
    monkeypatch.setattr(transport.time,"sleep",lambda *_:None)
    def action():
        calls.append(1)
        raise requests.exceptions.SSLError("EOF")
    with pytest.raises(requests.exceptions.SSLError):
        transport.retry(action)
    assert len(calls)==3


def test_local_release_requires_trusted_hash(tmp_path):
    path=tmp_path/"official-artifacts.zip"
    with zipfile.ZipFile(path,"w") as z:
        z.writestr("release_manifest.json",json.dumps({"suggested_tag":"lithiumscope-training_test","model_1":{},"model_2":{}}))
    with pytest.raises(ValueError,match="no coincide"):
        local_release(path,"0"*64)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    release,actual=local_release(path,digest)
    assert actual==digest and release.asset_digest=="sha256:"+digest
