"""One HTTP COG open for multiple nearby positions in exactly the same scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lithiumscope.experiments.acquisition.http_tiles import _interpolate, _sample_coordinates


def extract_many(href: str, positions: list[tuple[float, float]], *, size_m: float = 640,
                 pixels: int = 32, resampling: str = "bilinear") -> np.ndarray:
    import fsspec
    import tifffile

    if not positions:
        return np.empty((0, pixels, pixels), dtype=np.float32)
    with fsspec.open(href, "rb", block_size=32768).open() as source:
        with tifffile.TiffFile(source) as tif:
            page = tif.pages[0]
            if not page.is_tiled:
                raise ValueError("COG must be tiled")
            patches = []
            for lat, lon in positions:
                rows, cols = _sample_coordinates(page, lat, lon, size_m, pixels)
                patches.append(_interpolate(page, source, rows, cols, resampling))
            return np.stack(patches).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--href", required=True)
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("reflectance", "scl"), default="reflectance")
    args = parser.parse_args()
    positions = json.loads(args.positions.read_text(encoding="utf-8"))
    points = [(float(item["latitude"]), float(item["longitude"])) for item in positions]
    patches = extract_many(args.href, points,
                           resampling="nearest" if args.mode == "scl" else "bilinear")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, patches, allow_pickle=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
