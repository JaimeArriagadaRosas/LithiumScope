from __future__ import annotations

import re

_CHECK_IE_RE = re.compile(
    r"""^javascript\s*:\s*checkIE\(\s*['"](?P<path>[^'"]+)['"]""",
    re.IGNORECASE,
)


def resolve_download_href(
    href: str,
) -> str | None:
    value = str(href).strip()
    if not value:
        return None

    if not value.lower().startswith(
        "javascript:"
    ):
        return value

    match = _CHECK_IE_RE.match(value)
    if match is None:
        return None

    return match.group("path")
