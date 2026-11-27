"""Translate the declared imagery configuration into a provider contract."""
from .provider import SentinelConfig

def sentinel_config(config: dict) -> SentinelConfig:
    imagery = config["imagery"]
    return SentinelConfig(
        stac_url=str(imagery["stac_url"]),
        collection=str(imagery["collection"]),
        datetime=str(imagery["datetime"]),
        cloud_cover_max=float(imagery["cloud_cover_max"]),
        search_limit=int(imagery["search_limit"]),
        request_timeout_seconds=float(
            imagery.get("request_timeout_seconds", 60)
        ),
        max_retries=int(imagery.get("max_retries", 3)),
        scene_cache_decimals=int(
            imagery.get("scene_cache_decimals", 4)
        ),
        patch_size_m=float(imagery["patch_size_m"]),
        patch_pixels=int(imagery["patch_pixels"]),
        bands=tuple(str(value) for value in imagery["bands"]),
    )

