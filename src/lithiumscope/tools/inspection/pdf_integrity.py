"""Read-only PDF structure checks; no renderer, external requests or visual claims."""
from __future__ import annotations

from pathlib import Path
import re


def inspect_pdf(path: Path) -> dict:
    report = {"state": "INVALID", "page_count": None, "visual_review": False}
    try:
        with path.open("rb") as stream:
            if not stream.read(8).startswith(b"%PDF-"):
                return {**report, "reason": "header"}
            size = path.stat().st_size
            stream.seek(max(0, size - 8192))
            tail = stream.read()
            match = re.search(rb"startxref\s+(\d+)\s+%%EOF\s*$", tail)
            if not match:
                return {**report, "reason": "trailer"}
            offset = int(match.group(1))
            if not 8 <= offset < size:
                return {**report, "reason": "xref_offset"}
            stream.seek(offset)
            marker = stream.read(128)
            if not (marker.startswith(b"xref") or re.match(rb"\d+\s+\d+\s+obj\b", marker)):
                return {**report, "reason": "xref_marker"}
        try:
            from pypdf import PdfReader
        except ImportError:
            return {**report, "state": "SCREENED", "reason": "pypdf_unavailable"}
        with path.open("rb") as stream:
            reader = PdfReader(stream, strict=True)
            if reader.is_encrypted:
                return {**report, "reason": "encrypted"}
            count = len(reader.pages)
            if not count:
                return {**report, "reason": "no_pages"}
            for page in reader.pages:
                # Traverse page tree and decode content streams, not their text.
                content = page.get_contents()
                if content is not None:
                    content.get_data()
            return {**report, "state": "VERIFIED_STRUCTURE", "page_count": count}
    except Exception as exc:
        return {**report, "reason": type(exc).__name__}
