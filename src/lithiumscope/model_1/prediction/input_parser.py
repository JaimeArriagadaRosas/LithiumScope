from pathlib import Path
from lithiumscope.datasets.readers.tabular import load_data

def load_prediction_input(path:Path): return load_data(path)
