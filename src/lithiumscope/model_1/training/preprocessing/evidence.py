"""Inspect fitted TRAIN-only transformations without fitting on validation or TEST."""
from __future__ import annotations

import numpy as np

from lithiumscope.model_1.training.preprocessing.target_transform import unwrap_fitted_regressor


def preprocessing_evidence(estimator, x_train, x_test):
    pipeline = unwrap_fitted_regressor(estimator)
    if not hasattr(pipeline,"named_steps") or "preprocess" not in pipeline.named_steps:
        return None
    prep = pipeline.named_steps["preprocess"]
    names = list(map(str,prep.get_feature_names_out()))
    if len(names)!=len(set(names)):
        raise ValueError("Duplicate transformed feature names")
    # Read actual dimensions and fitted statistics, never fit on TEST.
    train_matrix=prep.transform(x_train)
    test_matrix=prep.transform(x_test)
    if train_matrix.shape!=(len(x_train),len(names)) or test_matrix.shape!=(len(x_test),len(names)):
        raise ValueError("Transformed TRAIN/TEST schema differs")
    stats={}
    for name, transformer, cols in prep.transformers_:
        if name=="remainder" or not len(cols):
            continue
        record={"columns":list(cols)}
        steps=getattr(transformer,"named_steps",{})
        for key,attr in (("imputer","statistics_"),("scaler","mean_"),("scaler","scale_"),("scaler","var_")):
            if key in steps and hasattr(steps[key],attr):
                values=np.asarray(getattr(steps[key],attr))
                record[f"{key}_{attr}"]=[float(v) if np.isfinite(v) else None for v in values] if values.dtype.kind in "fiu" else list(map(str,values))
        encoder=steps.get("onehot",transformer)
        if hasattr(encoder,"vocabulary_"):
            record["train_category_vocabulary"]=[sorted(v) for v in encoder.vocabulary_]
            record["unseen_test_counts"]={str(c):int(sum(str(v) not in known for v in x_test[c].dropna()))
                                           for c,known in zip(cols,encoder.vocabulary_,strict=True)}
        elif hasattr(encoder,"categories_"):
            record["train_category_vocabulary"]=[list(map(str,v)) for v in encoder.categories_]
        stats[name]=record
    return {"transformed_features":names,"feature_count":len(names),
            "train_rows":len(x_train),"test_rows":len(x_test),
            "train_all_missing_columns":[str(c) for c in x_train if x_train[c].isna().all()],
            "train_missing_counts":{str(c):int(x_train[c].isna().sum()) for c in x_train},
            "test_missing_counts":{str(c):int(x_test[c].isna().sum()) for c in x_test},
            "transformers":stats}
