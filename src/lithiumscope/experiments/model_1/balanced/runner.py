"""Second-stage M1 study: seven policies x three existing regression families."""
from __future__ import annotations

import argparse

import numpy as np

from lithiumscope.experiments.model_1.balanced.regimes import regime_fit
from lithiumscope.experiments.model_1.balanced.selection import ARMS,balanced_positions
from lithiumscope.experiments.model_1.targeted.fit import _base_fit,fit_arm
from lithiumscope.experiments.model_1.targeted.policies import Arm
from lithiumscope.experiments.model_1.targeted.runner import run_targeted


def fit_balanced(algorithm,arm,device,seed,params,x,y,groups,sources,numeric,categorical,validation):
    args=(algorithm,device,seed,params)
    if arm.name=="control":
        return fit_arm(algorithm,Arm("control"),*args[1:],x,y,groups,sources,numeric,categorical,validation)
    if arm.regime:
        return regime_fit(algorithm,arm,device,seed,params,x,y,groups,sources,numeric,categorical,validation)
    keep=balanced_positions(x,y,groups,sources,arm.cohort_target,seed)
    predict,info=_base_fit(algorithm,Arm("control"),device,seed,params,
        x.iloc[keep],y.iloc[keep],groups.iloc[keep],sources.iloc[keep],numeric,categorical)
    dropped=np.setdiff1d(np.arange(len(x)),keep)
    info.update(real_fit_rows=len(keep),original_train_rows=len(x),cohort_equivalent_target=arm.cohort_target,
                calibration_fits=0,extra_regime_fits=0,
                trace=[*info["trace"],*({"row":str(x.index[i]),"action":"train_only_subsample_exclusion"} for i in dropped)])
    # TRAIN partition diagnostics are on the full original TRAIN, not only retained rows.
    return np.maximum(predict(validation),0.),np.maximum(predict(x),0.),info


def run_balanced(run_id,**kwargs):
    return run_targeted(run_id,arms=ARMS,fit_function=fit_balanced,
                        campaign="model_1_balanced",**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run",required=True)
    args=parser.parse_args()
    run_balanced(args.previous_run)


if __name__=="__main__":
    main()
