"""Atomic generic output writers; callers own scientific interpretation."""
from __future__ import annotations
import json
from pathlib import Path
from uuid import uuid4


def _publish(path,write):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_json(path,payload):
    return _publish(path,lambda p:p.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8"))


def write_table(path,frame):
    return _publish(path,lambda p:frame.to_csv(p,index=False))
