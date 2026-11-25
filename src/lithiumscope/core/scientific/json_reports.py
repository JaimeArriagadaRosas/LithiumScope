from __future__ import annotations
import json
from pathlib import Path

def save_json_report(payload:dict,destination:Path)->Path:
    destination.parent.mkdir(parents=True,exist_ok=True); destination.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8"); return destination
