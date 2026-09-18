from pathlib import Path
import pandas as pd
from lithiumscope.core.config import load_config
from lithiumscope.model_2.steps.step_07_build_training_set import build_training_set

def load_model_2_training_frame(path:Path|None=None)->pd.DataFrame:
    cfg=load_config("model_2");training_cfg=cfg["training"];manifest_path=path or Path(training_cfg["manifest_path"])
    if not manifest_path.exists():raise FileNotFoundError(f"Model 2 manifest not found: {manifest_path}. It must contain one row per known sample and an image_path column.")
    return build_training_set(pd.read_csv(manifest_path),lithium_column=str(training_cfg["lithium_column"]))
