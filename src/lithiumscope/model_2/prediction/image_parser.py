from pathlib import Path
import pandas as pd
from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery
from lithiumscope.model_2.steps.step_04_image_preprocessing import preprocess_image
from lithiumscope.model_2.steps.step_05_spectral_features import extract_spectral_features
def image_to_feature_frame(path:Path)->pd.DataFrame:return pd.DataFrame([extract_spectral_features(preprocess_image(load_imagery(path)))])
