from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lithiumscope.tools.georoc_material_filter import (
    MaterialFilterResult,
    filter_whole_rock,
)
from lithiumscope.tools.georoc_normalizer import (
    normalize_georoc_frame,
)
from lithiumscope.tools.georoc_raw_reader import (
    read_georoc_raw,
)
from lithiumscope.tools.georoc_schema_profile import (
    profile_georoc_schema,
)


@dataclass(frozen=True)
class GeorocPostprocessResult:
    path: Path
    encoding: str
    rows_before_filter: int
    rows_after_filter: int
    rows_removed_by_material: int
    material_column: str
    materials_seen: tuple[str, ...]
    whole_rock_rows: int
    volcanic_glass_rows: int
    unknown_rows: int
    missing_rows: int


def process_georoc_text_export(
    raw_path: Path,
    destination: Path,
    *,
    profile_callback=None,
) -> GeorocPostprocessResult:
    raw = read_georoc_raw(raw_path)
    normalized = normalize_georoc_frame(
        raw.frame
    )
    if profile_callback is not None:
        profile_callback(
            profile_georoc_schema(
                normalized
            )
        )
    material: MaterialFilterResult = (
        filter_whole_rock(normalized)
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = destination.with_name(
        destination.name + ".normalized.part"
    )
    temporary.unlink(missing_ok=True)
    try:
        material.frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8",
        )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return GeorocPostprocessResult(
        path=destination,
        encoding=raw.encoding,
        rows_before_filter=material.rows_before,
        rows_after_filter=material.rows_after,
        rows_removed_by_material=material.rows_removed,
        material_column=material.material_column,
        materials_seen=material.materials_seen,
        whole_rock_rows=material.whole_rock_rows,
        volcanic_glass_rows=material.volcanic_glass_rows,
        unknown_rows=material.unknown_rows,
        missing_rows=material.missing_rows,
    )
