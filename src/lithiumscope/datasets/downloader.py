from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import requests
from lithiumscope.core.exceptions import DatasetError
from lithiumscope.core.logger import get_logger
from lithiumscope.core.paths import DATA_DIR
from lithiumscope.datasets.registry import DatasetSpec, get_dataset_spec
from lithiumscope.datasets.validator import validate_nonempty
logger=get_logger("datasets")

def _stream_download(url:str,destination:Path)->Path:
    destination.parent.mkdir(parents=True,exist_ok=True);logger.info("Downloading %s -> %s",url,destination)
    with requests.get(url,stream=True,timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:handle.write(chunk)
    return validate_nonempty(destination)

def _zenodo_manifest(record:int)->list[dict]:
    response=requests.get(f"https://zenodo.org/api/records/{record}",timeout=60);response.raise_for_status();files=response.json().get("files",[])
    if not files:raise DatasetError(f"Zenodo record {record} contains no downloadable files.")
    return files

def _zenodo_download(spec:DatasetSpec,destination_dir:Path)->Path:
    if spec.zenodo_record is None:raise DatasetError(f"Zenodo record missing for {spec.key}")
    destination_dir.mkdir(parents=True,exist_ok=True);manifest=_zenodo_manifest(spec.zenodo_record);selected=set(spec.zenodo_files);(destination_dir/"zenodo_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");downloaded=0
    for item in manifest:
        key=item.get("key") or item.get("filename")
        if not key or (selected and key not in selected):continue
        links=item.get("links",{});url=links.get("content") or links.get("self")
        if not url:continue
        target=destination_dir/key
        if target.exists() and target.stat().st_size>0:logger.info("Dataset file already exists: %s",target);downloaded+=1;continue
        _stream_download(url,target);downloaded+=1
    if downloaded==0:raise DatasetError(f"No files matched the configured Zenodo selection for {spec.key}")
    return validate_nonempty(destination_dir)

def ensure_dataset(key:str,allow_large:bool=False)->Path:
    spec=get_dataset_spec(key);base=DATA_DIR/"raw"/spec.model
    if spec.provider=="dynamic":raise DatasetError(f"{spec.key} is a dynamic imagery provider, not a static dataset. Use its scene acquisition pipeline when coordinates are available.")
    if spec.large and not allow_large:raise DatasetError(f"{spec.key} is marked as a large download. Re-run with allow_large=True after reviewing the source: {spec.source_url}")
    if spec.provider=="direct":
        destination=base/spec.destination_name
        if destination.exists() and destination.stat().st_size>0:return destination
        if not spec.download_url:raise DatasetError(f"No direct URL configured for {spec.key}")
        return _stream_download(spec.download_url,destination)
    if spec.provider=="zenodo":
        destination=base/spec.destination_name
        if destination.exists() and any(destination.iterdir()):return destination
        return _zenodo_download(spec,destination)
    raise DatasetError(f"Unsupported dataset provider: {spec.provider}")

def ensure_many(keys:Iterable[str],allow_large:bool=False)->list[Path]:return [ensure_dataset(key,allow_large=allow_large) for key in keys]
