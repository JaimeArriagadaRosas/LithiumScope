"""Prepare visualization artifacts before entering the PDF renderer."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import numpy as np
from lithiumscope.core.logging.session import get_logger

logger = get_logger("reporting.figures.decision")


def prepare_decision_previews(selected, identity: dict, destination: Path) -> dict[str, Path]:
    """Never guess the band order of a historical or custom patch."""
    bands = identity.get("bands")
    if not isinstance(bands, (list, tuple)) or not bands or selected.empty:
        return {}
    from PIL import Image
    from lithiumscope.reporting.figures.satellite import _rgb_preview

    previews = {}
    for _, row in selected.iterrows():
        case_id = str(row["case_id"])
        source = row.get("image")
        if not isinstance(source, (str, Path)) or not Path(source).is_file():
            continue
        try:
            rgb = _rgb_preview(Path(source), bands)
            if rgb is None:
                continue
            pixels = np.uint8(np.clip(rgb, 0, 1) * 255)
            destination.mkdir(parents=True, exist_ok=True)
            name = sha256(case_id.encode("utf-8")).hexdigest() + ".png"
            path = destination / name
            temporary = destination / f".{uuid4().hex}.png"
            try:
                Image.fromarray(pixels, mode="RGB").resize(
                    (192, 192), Image.Resampling.NEAREST
                ).save(temporary, format="PNG")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            previews[case_id] = path
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("Decision preview omitted case=%s: %s", case_id, exc)
            # The scientific result survives absent or unreadable illustrations.
            continue
    return previews
