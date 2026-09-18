from pathlib import Path
from lithiumscope.model_1.steps.step_01_load_data import load_data

def load_prediction_input(path:Path): return load_data(path)
