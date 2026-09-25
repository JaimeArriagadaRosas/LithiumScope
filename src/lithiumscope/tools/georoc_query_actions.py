from __future__ import annotations

import re

from lithiumscope.tools.georoc_query_html import norm
from lithiumscope.tools.georoc_query_models import Form

_POSTPAGE_RE = re.compile(
    r"""postpage\(\s*['"][^'"]+['"]\s*,\s*['"](?P<target>[^'"]+)['"]\s*,\s*['"](?P<track>[^'"]*)['"]\s*\)""",
    re.IGNORECASE,
)


def resolve_postpage_action(
    form: Form,
    label: str,
) -> tuple[str, str]:
    wanted = norm(label)

    for item in form.inputs:
        if item.kind not in {
            "submit",
            "button",
            "image",
        }:
            continue

        visible = norm(
            " ".join(
                (
                    item.value,
                    item.nearby_text,
                )
            )
        )
        if wanted not in visible:
            continue

        match = _POSTPAGE_RE.search(
            item.onclick
        )
        if match is None:
            raise RuntimeError(
                "GEOROC encontró el botón solicitado, "
                "pero su acción JavaScript no corresponde "
                "al patrón postpage esperado."
            )

        return (
            match.group("target"),
            match.group("track"),
        )

    raise RuntimeError(
        "GEOROC no expuso de forma inequívoca "
        f"la acción {label!r}."
    )
