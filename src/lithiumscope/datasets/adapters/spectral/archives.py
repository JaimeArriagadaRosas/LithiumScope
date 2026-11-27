"""Bounded, explicit ZIP-member reading; never extract an untrusted directory tree."""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from stat import S_ISLNK

def _safe_member(name):
    parts = PurePosixPath(name).parts
    return bool(parts) and not name.startswith("/") and chr(92) not in name and ":" not in name and ".." not in parts

def list_text_members(archive):
    with ZipFile(archive) as handle:
        return sorted(info.filename for info in handle.infolist()
                      if not info.is_dir() and _safe_member(info.filename)
                      and not S_ISLNK(info.external_attr >> 16)
                      and PurePosixPath(info.filename).suffix.lower() in {".txt", ".csv"})

@contextmanager
def selected_text_member(archive, member, *, maximum_bytes=20*1024*1024):
    if not _safe_member(member) or PurePosixPath(member).suffix.lower() not in {".txt", ".csv"}:
        raise ValueError("Unsafe or unsupported spectral archive member")
    if not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
        raise ValueError("Invalid member size limit")
    with ZipFile(archive) as handle:
        matches = [info for info in handle.infolist() if info.filename == member]
        if len(matches) != 1:
            raise ValueError("Archive member missing or ambiguous")
        info = matches[0]
        if info.is_dir() or S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1 or not 0 < info.file_size <= maximum_bytes:
            raise ValueError("Archive member is unsafe, encrypted or exceeds size policy")
        with TemporaryDirectory(prefix="lithiumscope-spectrum-") as folder:
            path = Path(folder) / PurePosixPath(member).name
            with handle.open(info) as source, path.open("wb") as output:
                remaining = maximum_bytes
                while block := source.read(min(1024*1024, remaining+1)):
                    remaining -= len(block)
                    if remaining < 0:
                        raise ValueError("Archive member exceeds size policy")
                    output.write(block)
            if path.stat().st_size != info.file_size:
                raise ValueError("Archive member length mismatch")
            yield path
