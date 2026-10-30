"""Isolated raster read worker: allows parent to enforce a hard timeout."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--href", required=True)
    parser.add_argument("--lat", required=True, type=float)
    parser.add_argument("--lon", required=True, type=float)
    parser.add_argument("--size", required=True, type=float)
    parser.add_argument("--pixels", required=True, type=int)
    parser.add_argument("--mode", choices=["reflectance","scl"], default="reflectance")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from lithiumscope.model_2.data.sentinel2 import _read_patch
    if args.mode == "reflectance":
        array = _read_patch(args.href, args.lat, args.lon, args.size, args.pixels)
    else:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import transform
        from rasterio.windows import from_bounds
        with rasterio.Env(
            AWS_NO_SIGN_REQUEST="YES",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_CONNECTTIMEOUT="10",
            GDAL_HTTP_TIMEOUT="15",
            GDAL_HTTP_MAX_RETRY="1",
        ):
            with rasterio.open(args.href) as source:
                x, y = transform("EPSG:4326",source.crs,[args.lon],[args.lat])
                half = args.size/2
                window = from_bounds(x[0]-half,y[0]-half,x[0]+half,y[0]+half,source.transform)
                values = source.read(1,window=window,out_shape=(args.pixels,args.pixels),
                                     boundless=True,masked=True,resampling=Resampling.nearest)
                array=np.asarray(values.filled(0),dtype=np.float32)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(args.output,array)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
