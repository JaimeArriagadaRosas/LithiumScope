"""Scientific CLI entrypoints are grouped by purpose, with traceable relocation."""
import json
from pathlib import Path

def test_scientific_script_relocation_and_directory_limits():
    root=Path(__file__).resolve().parents[2]
    mapping=json.loads((root/"docs/integral-refactor/scientific-script-relocation.json").read_text())
    assert len(mapping)==15
    assert all((root/new).is_file() and not (root/old).exists() for old,new in mapping.items())
    folder=root/"scripts/scientific"
    for directory in (folder,*[p for p in folder.rglob("*") if p.is_dir()]):
        assert len(list(directory.glob("*.py")))<=20, directory
