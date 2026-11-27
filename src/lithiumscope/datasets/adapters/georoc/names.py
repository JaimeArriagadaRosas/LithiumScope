"""Shared GEOROC field-name normalization."""
import re


def norm(value: str) -> str:
    return re.sub(
        r"[^A-Z0-9]+",
        "",
        str(value).upper(),
    )
