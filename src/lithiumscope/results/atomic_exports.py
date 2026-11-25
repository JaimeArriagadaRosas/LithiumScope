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


def write_table(path,frame,*,index=False):
    return _publish(path,lambda p:frame.to_csv(p,index=index))


def write_text(path,text):
    return _publish(path,lambda p:p.write_text(text,encoding="utf-8"))


def write_legacy_evidence_json(path,payload):
    """Preserve legacy string fallback while replacing nonfinite values with null."""
    from math import isfinite
    from numbers import Real, Integral
    def normalize(value):
        if isinstance(value,dict):
            return {str(k):normalize(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):
            return [normalize(v) for v in value]
        if value is None or isinstance(value,(str,bool)):
            return value
        if isinstance(value,Integral):
            return int(value)
        if isinstance(value,Real):
            return float(value) if isfinite(value) else None
        return str(value)
    return write_json(path,normalize(payload))

