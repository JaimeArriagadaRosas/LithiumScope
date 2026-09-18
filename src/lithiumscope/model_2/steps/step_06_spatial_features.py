def spatial_features(latitude:float|None,longitude:float|None)->dict[str,float]:
    features={}
    if latitude is not None:features["latitude"]=float(latitude)
    if longitude is not None:features["longitude"]=float(longitude)
    return features
