"""Restore only explicitly mapped, hash-verified evidence from the private backup."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
MAP=ROOT/"docs/scientific/continuity/2026-11-26/restore_map.json"


def restore(archive,root=ROOT,*,apply=False,mapping=None):
    root=Path(root).resolve()
    mapping=json.loads(MAP.read_text(encoding="utf-8")) if mapping is None else mapping
    records=[]
    with ZipFile(archive) as bundle:
        payloads=[]
        for name,item in mapping.items():
            path=root/item["destination"]
            if not path.resolve().is_relative_to(root/"results") or path.is_symlink():
                raise ValueError("Evidence destination outside results")
            raw=bundle.read("originales/"+name)
            if sha256(raw).hexdigest()!=item["sha256"]:
                raise ValueError("Backup checksum mismatch: "+name)
            state="ready"
            if path.exists():
                state="verified" if sha256(path.read_bytes()).hexdigest()==item["sha256"] else "conflict"
            records.append({"original":name,"destination":item["destination"],"state":state})
            payloads.append((path,raw))
        if apply:
            for record,(path,raw) in zip(records,payloads,strict=True):
                if record["state"]!="ready":
                    continue
                path.parent.mkdir(parents=True,exist_ok=True)
                try:
                    with path.open("xb") as stream:
                        stream.write(raw)
                except FileExistsError:
                    record["state"]="conflict"
                    continue
                record["state"]="restored"
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args()
    rows=restore(args.archive,apply=args.apply)
    for row in rows:
        print(row["state"].upper(),row["destination"])
    if any(r["state"]=="conflict" for r in rows):
        print("Hay conflictos: los originales existentes no se sobrescribieron.")
    if not args.apply:
        print("Vista previa; añada --apply para restaurar.")
    return int(any(r["state"]=="conflict" for r in rows))


if __name__=="__main__":
    raise SystemExit(main())
