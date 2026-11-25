"""Read small COG windows via HTTP Range, without GDAL network I/O.

Uses pixel-center bilinear sampling for reflectance and nearest-neighbour for SCL.
Source raster values remain in their original training-time units (digital numbers).
"""
from __future__ import annotations

import numpy as np


def _epsg(page) -> int:
    keys = page.tags["GeoKeyDirectoryTag"].value
    for offset in range(4, len(keys), 4):
        key, location, count, value = keys[offset:offset + 4]
        if key == 3072 and location == 0 and count == 1:
            return int(value)
    raise ValueError("GeoTIFF lacks projected EPSG code")


def _sample_coordinates(page, latitude: float, longitude: float,
                        size_m: float, pixels: int) -> tuple[np.ndarray, np.ndarray]:
    from rasterio.warp import transform

    scale = page.tags["ModelPixelScaleTag"].value
    tie = page.tags["ModelTiepointTag"].value
    if scale[0] <= 0 or scale[1] <= 0:
        raise ValueError("Expected north-up raster with positive pixel scale")
    x, y = transform("EPSG:4326", f"EPSG:{_epsg(page)}", [longitude], [latitude])
    step = size_m / pixels
    output_x = x[0] - size_m / 2 + (np.arange(pixels) + 0.5) * step
    output_y = y[0] + size_m / 2 - (np.arange(pixels) + 0.5) * step
    # Pixel centres have integer indices, not pixel corners.
    columns = (output_x - (tie[3] - tie[0] * scale[0])) / scale[0] - 0.5
    rows = ((tie[4] + tie[1] * scale[1]) - output_y) / scale[1] - 0.5
    return rows, columns


def _read_samples(page, source, row_indices: np.ndarray,
                  col_indices: np.ndarray) -> np.ndarray:
    """Decode only necessary native tiles; NaN for nodata and outside image."""
    result = np.full((len(row_indices), len(col_indices)), np.nan, dtype=np.float32)
    tiles_across = (page.imagewidth + page.tilewidth - 1) // page.tilewidth
    cache: dict[int, np.ndarray] = {}
    nodata = page.nodata
    for r, row in enumerate(row_indices):
        if not 0 <= row < page.imagelength:
            continue
        for c, col in enumerate(col_indices):
            if not 0 <= col < page.imagewidth:
                continue
            idx = (row // page.tilelength) * tiles_across + col // page.tilewidth
            if idx not in cache:
                source.seek(page.dataoffsets[idx])
                raw = source.read(page.databytecounts[idx])
                if len(raw) != page.databytecounts[idx]:
                    raise OSError("Incomplete HTTP range response")
                decoded, _, _ = page.decode(raw, idx)
                cache[idx] = np.asarray(decoded[0, :, :, 0])
            value = float(cache[idx][row % page.tilelength, col % page.tilewidth])
            if nodata is None or value != float(nodata):
                result[r, c] = value
    return result


def _interpolate(page, source, fractional_rows: np.ndarray,
                 fractional_cols: np.ndarray, method: str) -> np.ndarray:
    if method == "nearest":
        row_idx = np.floor(fractional_rows + 0.5).astype(int)
        col_idx = np.floor(fractional_cols + 0.5).astype(int)
        return _read_samples(page, source, row_idx, col_idx)
    if method != "bilinear":
        raise ValueError(f"Unsupported resampling: {method}")

    bottom = np.floor(fractional_rows).astype(int)
    left = np.floor(fractional_cols).astype(int)
    dy = fractional_rows - bottom
    dx = fractional_cols - left
    row_idx = np.unique(np.concatenate([bottom, bottom + 1]))
    col_idx = np.unique(np.concatenate([left, left + 1]))
    native = _read_samples(page, source, row_idx, col_idx)
    r0 = np.searchsorted(row_idx, bottom)
    r1 = np.searchsorted(row_idx, bottom + 1)
    c0 = np.searchsorted(col_idx, left)
    c1 = np.searchsorted(col_idx, left + 1)
    corners = [
        (native[np.ix_(r0, c0)], (1 - dy)[:, None] * (1 - dx)[None, :]),
        (native[np.ix_(r0, c1)], (1 - dy)[:, None] * dx[None, :]),
        (native[np.ix_(r1, c0)], dy[:, None] * (1 - dx)[None, :]),
        (native[np.ix_(r1, c1)], dy[:, None] * dx[None, :]),
    ]
    numerator = np.zeros((len(dy), len(dx)), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    for values, weights in corners:
        valid = np.isfinite(values)
        numerator += np.where(valid, values, 0) * weights
        denominator += valid * weights
    return np.divide(numerator, denominator,
                     out=np.full(numerator.shape, np.nan), where=denominator > 0).astype(np.float32)


def extract_http_patch(href: str, latitude: float, longitude: float,
                       size_m: float = 640, pixels: int = 32,
                       *, resampling: str = "bilinear") -> np.ndarray:
    """Read a single 32x32-band patch, equivalent in sampling geometry to GDAL."""
    import fsspec
    import tifffile

    if pixels <= 0 or size_m <= 0:
        raise ValueError("Positive pixels and size_m required")
    with fsspec.open(href, "rb", block_size=32768).open() as source:
        with tifffile.TiffFile(source) as tif:
            page = tif.pages[0]
            if not page.is_tiled:
                raise ValueError("HTTP range reader requires tiled GeoTIFF")
            row_coords, col_coords = _sample_coordinates(
                page, latitude, longitude, size_m, pixels)
            return _interpolate(page, source, row_coords, col_coords, resampling)
