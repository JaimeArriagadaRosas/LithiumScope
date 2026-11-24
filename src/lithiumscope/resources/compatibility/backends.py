"""Small actual backend probes; CUDA build flags alone are not evidence."""
from __future__ import annotations

import importlib.util
import json


def probe_xgboost_cuda() -> tuple[bool | None, str | None]:
    if importlib.util.find_spec("xgboost") is None:
        return None, "XGBoost no está instalado."
    try:
        import numpy as np
        import xgboost as xgb
        value = xgb.build_info().get("USE_CUDA")
        if str(value).lower() not in {"true", "1", "yes", "on"}:
            return False, "XGBoost no fue compilado con CUDA."
        model = xgb.XGBRegressor(n_estimators=1, max_depth=1, tree_method="hist",
                                 device="cuda", verbosity=0, n_jobs=1)
        model.fit(np.arange(4, dtype=float).reshape(-1, 1), np.arange(4, dtype=float))
        device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not str(device).startswith("cuda"):
            return False, "XGBoost retornó a CPU durante la comprobación."
        return True, None
    except Exception as exc:
        return False, f"XGBoost CUDA falló: {exc}"


def probe_catboost_gpu() -> tuple[bool | None, str | None]:
    if importlib.util.find_spec("catboost") is None:
        return None, "CatBoost no está instalado."
    try:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=1, depth=2, task_type="GPU", devices="0",
                                  thread_count=1, verbose=False, allow_writing_files=False)
        model.fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 1.0, 2.0, 3.0])
        return True, None
    except Exception as exc:
        return False, f"CatBoost GPU falló: {exc}"


def xgboost_device_parameters(device) -> dict[str, str]:
    return {"device": "cuda" if device.xgboost_cuda_available else "cpu", "tree_method": "hist"}
