from pathlib import Path
import pandas as pd
from lithiumscope.core.logger import get_logger
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.model_1.pipeline import prepare_prediction_data
from lithiumscope.model_1.prediction.input_parser import load_prediction_input
from lithiumscope.persistence.load_model import load_latest_model
logger=get_logger("model_1.predictor")

def predict_model_1(path:Path)->tuple[pd.DataFrame,Path,Path]:
    bundle,model_path=load_latest_model("model_1"); frame=load_prediction_input(path); x=prepare_prediction_data(frame,bundle["algorithm"],bundle["schema"])
    output=frame.copy(); output["Li_icpms_predicted"]=bundle["estimator"].predict(x)
    destination=RESULTS_DIR/"model_1"/"predictions"/f"{path.stem}_predictions.csv"; destination.parent.mkdir(parents=True,exist_ok=True); output.to_csv(destination,index=False); logger.info("Model 1 predictions written to %s",destination); return output,destination,model_path
