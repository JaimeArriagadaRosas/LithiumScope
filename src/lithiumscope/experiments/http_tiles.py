"""HTTP Range / TIFF tile extraction, bypassing remote GDAL VSI on Windows."""
from __future__ import annotations
import numpy as np


def _epsg(page):
    keys=page.tags["GeoKeyDirectoryTag"].value
    for offset in range(4, len(keys), 4):
        key, location, count, value = keys[offset:offset+4]
        if key == 3072 and location == 0 and count == 1:
            return int(value)
    raise ValueError("GeoTIFF lacks projected EPSG code")


def extract_http_patch(href: str, latitude: float, longitude: float,
                       size_m: float = 640, pixels: int = 32) -> np.ndarray:
    """Read only needed native COG tiles using HTTP range, then sample to 20m grid."""
    import fsspec
    import tifffile
    from rasterio.warp import transform
    if pixels <= 0 or size_m <= 0:
        raise ValueError("Positive pixels and size_m required")
    with fsspec.open(href, "rb", block_size=32768).open() as source:
        with tifffile.TiffFile(source) as tif:
            page=tif.pages[0]
            if not page.is_tiled:
                raise ValueError("HTTP range reader requires tiled GeoTIFF")
            scale=page.tags["ModelPixelScaleTag"].value
            tie=page.tags["ModelTiepointTag"].value
            epsg=_epsg(page)
            x,y=transform("EPSG:4326",f"EPSG:{epsg}",[longitude],[latitude])
            # Geographic centre of each requested 20m output pixel.
            step=size_m / pixels
            target_x=x[0]-size_m/2+(np.arange(pixels)+.5)*step
            target_y=y[0]+size_m/2-(np.arange(pixels)+.5)*step
            columns=np.floor((target_x-(tie[3]-tie[0]*scale[0]))/scale[0]).astype(int)
            rows=np.floor(((tie[4]+tie[1]*scale[1])-target_y)/scale[1]).astype(int)
            result=np.full((pixels,pixels),np.nan,dtype=np.float32)
            tiles_across=(page.imagewidth+page.tilewidth-1)//page.tilewidth
            cache={}
            for r, row in enumerate(rows):
                if not 0 <= row < page.imagelength:
                    continue
                for c, col in enumerate(columns):
                    if not 0 <= col < page.imagewidth:
                        continue
                    tile_row=row//page.tilelength
                    tile_col=col//page.tilewidth
                    idx=tile_row*tiles_across+tile_col
                    if idx not in cache:
                        source.seek(page.dataoffsets[idx])
                        raw=source.read(page.databytecounts[idx])
                        decoded,_,_=page.decode(raw,idx)
                        cache[idx]=decoded[0,:,:,0]
                    result[r,c]=cache[idx][row%page.tilelength,col%page.tilewidth]
            if page.nodata is not None:
                result[result==float(page.nodata)]=np.nan
            return result
