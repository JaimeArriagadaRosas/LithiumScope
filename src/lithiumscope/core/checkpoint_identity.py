"""Bind resumable folds and Optuna studies to ordered input and evaluation identity."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from lithiumscope.core.reproducibility import canonical_json_hash, _package_versions


def _data_identity(values):
    if values is None:
        return None
    frame=values if isinstance(values,(pd.Series,pd.DataFrame)) else pd.Series(values)
    hashes=pd.util.hash_pandas_object(frame,index=True).to_numpy(dtype="<u8")
    return {"hash":sha256(hashes.tobytes()).hexdigest(),
            "columns":list(map(str,frame.columns)) if isinstance(frame,pd.DataFrame) else [str(frame.name)],
            "dtypes":list(map(str,frame.dtypes)) if isinstance(frame,pd.DataFrame) else [str(frame.dtype)]}


def evaluation_identity(root, *, x, y, groups, splits, policy, extra=None, model_group=None):
    parent=Path(root).parent/"run.json"
    run_signature=None
    if parent.is_file():
        manifest=json.loads(parent.read_text(encoding="utf-8"))
        run_signature=manifest.get("summary",{}).get("training_signature")
        if not run_signature:
            raise ValueError("Training run lacks an evaluation signature")
    code_signature=None
    if run_signature is None and model_group is not None:
        from lithiumscope.core.run_resume import _scientific_code_fingerprint
        code_signature=_scientific_code_fingerprint(model_group)
    partitions=[{"train":np.asarray(tr).tolist(),"test":np.asarray(te).tolist()} for tr,te in splits]
    return canonical_json_hash({"version":1,"run_signature":run_signature,"standalone_code_signature":code_signature,"python":sys.version.split()[0],
        "packages":_package_versions(),"predictors":_data_identity(x),"target":_data_identity(y),
        "groups":_data_identity(groups),"partitions":partitions,"policy":policy,"extra":extra})
