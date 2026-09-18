import numpy as np
def extract_spectral_features(array:np.ndarray)->dict[str,float]:
    features={}
    for index,band in enumerate(array,start=1):
        features[f"band_{index:02d}_mean"]=float(np.mean(band));features[f"band_{index:02d}_std"]=float(np.std(band));features[f"band_{index:02d}_p10"]=float(np.percentile(band,10));features[f"band_{index:02d}_p50"]=float(np.percentile(band,50));features[f"band_{index:02d}_p90"]=float(np.percentile(band,90))
    return features
