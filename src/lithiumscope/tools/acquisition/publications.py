"""Bounded publication retrieval; metadata is not proof of sample-level methods."""
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import requests

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger


def fetch_public_pdf(url, destination, *, acquire=False, request=requests.get, maximum_bytes=64*1024*1024):
    destination=Path(destination)
    if maximum_bytes<=0:
        raise ValueError("Invalid PDF byte limit")
    if destination.is_file():
        with destination.open("rb") as stream:
            if stream.read(5)!=b"%PDF-":
                raise ValueError("Cached publication is not a PDF")
        if destination.stat().st_size>maximum_bytes:
            raise ValueError("Cached publication exceeds byte policy")
        return {"state":"cached","sha256":file_sha256(destination)}
    if not acquire:
        return {"state":"preflight","url":url}
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(f".{destination.name}.{uuid4().hex}.part")
    try:
        with request(url,timeout=(10,40),stream=True) as response:
            response.raise_for_status()
            total=0
            with temporary.open("wb") as stream:
                for block in response.iter_content(1024*1024):
                    total+=len(block)
                    if total>maximum_bytes:
                        raise ValueError("Publication exceeds byte policy")
                    stream.write(block)
        with temporary.open("rb") as stream:
            if stream.read(5)!=b"%PDF-":
                raise ValueError("Response is not a PDF")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    digest=file_sha256(destination)
    get_logger("acquisition.publications").info("Downloaded publication file=%s sha256=%s",destination.name,digest)
    return {"state":"downloaded","sha256":digest}


def lookup_doi(doi, *, acquire=False, request=requests.get):
    doi=str(doi).strip().lower().removeprefix("https://doi.org/").removeprefix("doi:")
    if not doi.startswith("10.") or "/" not in doi or any(c.isspace() for c in doi):
        raise ValueError("Invalid DOI")
    if not acquire:
        return {"state":"preflight","doi":doi}
    response=request("https://api.openalex.org/works/https://doi.org/"+quote(doi,safe="/"),timeout=16)
    response.raise_for_status()
    data=response.json()
    if not isinstance(data,dict):
        raise ValueError("Invalid OpenAlex response")
    returned=str(data.get("doi") or "").lower().removeprefix("https://doi.org/")
    if returned!=doi:
        raise ValueError("OpenAlex DOI identity mismatch")
    return data
