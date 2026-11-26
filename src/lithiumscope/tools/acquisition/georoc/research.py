"""Bounded acquisition of registered public research partitions; never train or merge."""
from __future__ import annotations

import json
from pathlib import Path
import re
from uuid import uuid4

import requests

from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.management.research_sources import ResearchPartition
from lithiumscope.tools.acquisition.georoc.dataverse import fetch_partition

METADATA = "https://data.goettingen-research-online.de/api/datasets/:persistentId/"
MAX_BYTES = 300_000_000


def discover(spec, *, request=requests.get):
    if spec.partitions:
        return spec.partitions
    response = request(METADATA, params={"persistentId": "doi:"+spec.doi}, timeout=(15,60))
    response.raise_for_status()
    data = response.json()["data"]["latestVersion"]
    if data["versionState"] != "RELEASED":
        raise ValueError("Only published Dataverse versions are accepted")
    pattern = r"^\d{4}-\d{2}-2JETOA_(" + "|".join(map(re.escape,spec.rock_types)) + r")(?:_part\d+)?\.csv$"
    chosen = []
    for entry in data["files"]:
        file = entry["dataFile"]
        if re.fullmatch(pattern, file["filename"], re.IGNORECASE):
            chosen.append((ResearchPartition(int(file["id"]), file["filename"]), int(file["filesize"])))
    if not chosen or not all(any(re.search("_"+rock+r"(?:_part\d+)?\.csv$", p.filename,re.I)
                                 for p,_ in chosen) for rock in spec.rock_types):
        raise ValueError("Published rock-type partitions incomplete")
    if sum(size for _,size in chosen) > MAX_BYTES:
        raise ValueError("Research source exceeds configured 300 MB acquisition budget")
    return tuple(sorted((p for p,_ in chosen),key=lambda p:p.filename))


def _publish(path, payload):
    temp = path.with_name("."+path.name+"."+uuid4().hex+".tmp")
    try:
        temp.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def acquire(spec, root, *, downloader=fetch_partition):
    root = Path(root)/spec.key
    root.mkdir(parents=True,exist_ok=True)
    receipt = root/"acquisition.json"
    plan_path = root/"partition_plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["doi"] != spec.doi:
            raise ValueError("Research source DOI mismatch")
        partitions = tuple(ResearchPartition(**p) for p in plan["partitions"])
    else:
        partitions = discover(spec)
        _publish(plan_path,{"doi":spec.doi,"partitions":[vars(p) for p in partitions]})
    if spec.partitions and partitions != spec.partitions:
        raise ValueError("Pinned partitions differ")
    old = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
    if old and (old.get("doi") != spec.doi or old.get("plan_sha256") != file_sha256(plan_path)):
        raise ValueError("Acquisition receipt does not match plan")
    known = {p["identifier"]:p for p in old.get("files",[])}
    records = []
    for partition in partitions:
        name = partition.filename
        if Path(name).name != name or not name.lower().endswith(".csv") or partition.identifier<=0:
            raise ValueError("Invalid research partition")
        path = root/name
        expected = known.get(partition.identifier,{}).get("sha256")
        if path.is_file() and expected and file_sha256(path) != expected:
            raise ValueError("Cached research partition changed")
        if not path.is_file():
            print(f"Descargando {spec.key}: {name}",flush=True)
        result = downloader(partition.identifier,path,expected_sha256=expected)
        records.append({"identifier":partition.identifier,"filename":name,"sha256":result["sha256"],
                        "state":result["state"],"bytes":path.stat().st_size})
        _publish(receipt,{"doi":spec.doi,"plan_sha256":file_sha256(plan_path),"files":records,
                          "state":"complete" if len(records)==len(partitions) else "partial"})
    return tuple(root/p.filename for p in partitions), receipt
