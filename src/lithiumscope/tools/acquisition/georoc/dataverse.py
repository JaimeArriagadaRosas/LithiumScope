"""Download explicitly selected published GEOROC Dataverse partitions atomically."""
from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import requests
from lithiumscope.core.hashing import file_sha256

BASE="https://data.goettingen-research-online.de/api/access/datafile/"


def fetch_partition(identifier:int,destination:Path,*,expected_sha256=None,timeout=75,request=requests.get):
    if type(identifier) is not int or identifier<=0 or timeout<=0:
        raise ValueError("Invalid Dataverse acquisition request")
    destination=Path(destination)
    if destination.is_file():
        digest=file_sha256(destination)
        if expected_sha256 and digest!=expected_sha256:
            raise ValueError("Existing partition hash differs; refusing overwrite")
        if destination.stat().st_size<1000:
            raise ValueError("Existing partition is incomplete; review manually")
        return {"path":str(destination),"sha256":digest,"state":"cached","verified_against_declared_hash":bool(expected_sha256)}
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with request(BASE+str(identifier),stream=True,timeout=(min(timeout,15),timeout)) as response:
            response.raise_for_status()
            size=0
            with temporary.open("xb") as handle:
                for chunk in response.iter_content(1024*1024):
                    if chunk:
                        size+=len(chunk)
                        handle.write(chunk)
            declared=response.headers.get("content-length")
            if size<1000 or (declared and not response.headers.get("content-encoding") and size!=int(declared)):
                raise ValueError("Incomplete partition transfer")
        digest=file_sha256(temporary)
        if expected_sha256 and digest!=expected_sha256:
            raise ValueError("Downloaded partition hash differs")
        # Exclusive publication: never replace a peer's existing download.
        import os
        os.link(temporary,destination)
        return {"path":str(destination),"sha256":digest,"state":"downloaded","verified_against_declared_hash":bool(expected_sha256)}
    finally:
        temporary.unlink(missing_ok=True)
