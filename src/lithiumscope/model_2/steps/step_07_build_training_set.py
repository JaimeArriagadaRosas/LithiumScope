from pathlib import Path
import pandas as pd
from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery
from lithiumscope.model_2.steps.step_04_image_preprocessing import preprocess_image
from lithiumscope.model_2.steps.step_05_spectral_features import extract_spectral_features

def build_training_set(manifest:pd.DataFrame,lithium_column:str="Li_icpms",image_path_column:str="image_path")->pd.DataFrame:
    if lithium_column not in manifest.columns:raise ValueError(f"Missing lithium column: {lithium_column}")
    if image_path_column not in manifest.columns:raise ValueError(f"Missing image path column: {image_path_column}")
    rows=[]
    for _,sample in manifest.iterrows():
        row=extract_spectral_features(preprocess_image(load_imagery(Path(sample[image_path_column]))));row[lithium_column]=float(sample[lithium_column]);rows.append(row)
    if not rows:raise ValueError("No image/sample pairs were available to build Model 2.")
    return pd.DataFrame(rows)
