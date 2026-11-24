from pathlib import Path
import pandas as pd
def match_samples_to_imagery(samples:pd.DataFrame,imagery_directory:Path,sample_id_column:str="sample_id")->pd.DataFrame:
    if sample_id_column not in samples.columns:raise ValueError(f"Missing sample id column: {sample_id_column}")
    image_index={p.stem:p for p in imagery_directory.iterdir() if p.suffix.lower() in {".npy",".tif",".tiff"}}
    result=samples.copy();result["image_path"]=result[sample_id_column].astype(str).map(image_index);return result.dropna(subset=["image_path"]).copy()
