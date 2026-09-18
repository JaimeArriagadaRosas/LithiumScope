from pathlib import Path
import time
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from lithiumscope.core.config import load_config
from lithiumscope.core.device import DeviceInfo
from lithiumscope.core.logger import get_logger
from lithiumscope.model_2.evaluation.metrics import classification_metrics
from lithiumscope.model_2.pipeline import load_model_2_training_frame
from lithiumscope.model_2.training.prospectivity_model import create_model
from lithiumscope.persistence.save_model import save_model_bundle
logger=get_logger("model_2.trainer")

def train_model_2(manifest_path:Path|None,device:DeviceInfo)->dict:
    cfg=load_config("model_2");model_cfg=cfg["model"];q=float(model_cfg["high_lithium_quantile"]);li=str(cfg["training"]["lithium_column"]);seed=int(model_cfg["random_seed"]);started=time.perf_counter();frame=load_model_2_training_frame(manifest_path);threshold=float(frame[li].quantile(q));y=(frame[li]>=threshold).astype(int);x=frame.drop(columns=[li])
    if y.nunique()<2:raise ValueError("Model 2 requires both high- and lower-lithium reference samples.")
    estimator=create_model(random_seed=seed);folds=max(2,min(5,int(y.value_counts().min())));cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed);probabilities=cross_val_predict(estimator,x,y,cv=cv,method="predict_proba",n_jobs=1)[:,1];metrics=classification_metrics(y.to_numpy(),probabilities);estimator.fit(x,y)
    report={"model_group":"model_2","algorithm":"random_forest","target_definition":f"Li_icpms >= q{q:.2f} ({threshold:.6f} ppm)","samples":int(len(frame)),"features":list(x.columns),"metrics":metrics,"device":device.accelerator,"device_name":device.name,"elapsed_seconds":round(time.perf_counter()-started,3),"scope_note":"Experimental prioritization baseline over pre-extracted image patches; not a probability of an economically viable deposit."}
    bundle={"estimator":estimator,"features":list(x.columns),"threshold_ppm":threshold,"target_quantile":q};model_path,metadata_path=save_model_bundle("model_2","prospectivity",bundle,report);report["model_path"]=str(model_path);report["metadata_path"]=str(metadata_path);logger.info("Model 2 training complete: %s",report);return report
