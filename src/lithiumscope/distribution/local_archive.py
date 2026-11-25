"""Explicit local ZIP recovery; verify trusted digest before deserialization."""
from __future__ import annotations

import json
from pathlib import Path
import re
import zipfile

from lithiumscope.core.hashing import file_sha256
from lithiumscope.distribution.github_releases import ModelRelease, _github_config


def local_release(path, expected_digest):
    path=Path(path)
    cfg=_github_config()
    digest=expected_digest.strip().lower().removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}",digest):
        raise ValueError("Se requiere SHA-256 confiable publicado o comprobado por el propietario")
    if not path.is_file() or path.stat().st_size>float(cfg.get("max_asset_size_mb",2048))*1048576:
        raise ValueError("ZIP local ausente o demasiado grande")
    if file_sha256(path)!=digest:
        raise ValueError("SHA-256 local no coincide; no se instalará")
    with zipfile.ZipFile(path) as archive:
        manifests=[]
        for member in archive.infolist():
            if "/" not in member.filename and member.filename.endswith(".json") and member.file_size<=1024*1024:
                value=json.loads(archive.read(member))
                if isinstance(value,dict) and {"suggested_tag","model_1","model_2"}<=set(value):
                    manifests.append(value)
        if len(manifests)!=1:
            raise ValueError("Se requiere un manifest inequívoco de release")
    tag=str(manifests[0]["suggested_tag"])
    if not tag.startswith(cfg["tag_prefix"]) or not re.fullmatch(r"[A-Za-z0-9_.-]+",tag):
        raise ValueError("Tag local incompatible")
    return ModelRelease(tag,tag,None,f"https://github.com/{cfg['repository']}/releases/tag/{tag}",
                        path.name,"",path.stat().st_size,"sha256:"+digest),digest
