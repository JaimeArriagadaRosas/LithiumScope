import numpy as np
def preprocess_image(array:np.ndarray,normalize_per_band:bool=True)->np.ndarray:
    result=np.asarray(array,dtype=np.float32).copy();result[~np.isfinite(result)]=np.nan
    for i in range(result.shape[0]):
        band=result[i];finite=band[np.isfinite(band)];fill=float(np.median(finite)) if finite.size else 0.0;band=np.nan_to_num(band,nan=fill,posinf=fill,neginf=fill)
        if normalize_per_band:
            mean=float(band.mean());std=float(band.std());band=(band-mean)/std if std>0 else band
        result[i]=band
    return result
