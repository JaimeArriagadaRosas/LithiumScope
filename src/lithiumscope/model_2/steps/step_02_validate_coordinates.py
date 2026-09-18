import pandas as pd
LATITUDE_CANDIDATES=("Latitude (Y)","Latitude","latitude","lat")
LONGITUDE_CANDIDATES=("Longitude (X)","Logintude (X)","Longitude","longitude","lon","lng")
def _resolve(frame,candidates):
    normalized={str(c).strip().lower():c for c in frame.columns}
    for candidate in candidates:
        match=normalized.get(candidate.lower())
        if match is not None:return match
    return None
def validate_coordinates(frame:pd.DataFrame)->tuple[str,str]:
    lat=_resolve(frame,LATITUDE_CANDIDATES);lon=_resolve(frame,LONGITUDE_CANDIDATES)
    if lat is None or lon is None:raise ValueError("Latitude/longitude columns are required for spatial matching.")
    return lat,lon
