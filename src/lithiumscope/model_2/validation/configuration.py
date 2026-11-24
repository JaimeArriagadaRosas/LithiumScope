"""Validate the effective canonical spatial M2 protocol before opening a training run."""
from __future__ import annotations

import math

from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.model_2.preparation.features.spectral import DEFAULT_BAND_NAMES
from lithiumscope.model_2.training.factory import ALGORITHMS
from lithiumscope.model_2.training.optimization.search_policy import parameter_space_for
from lithiumscope.model_2.training.calibration.policy import calibration_policy


def _require(condition, message):
    if not condition:
        raise ValueError("M2 configuration: "+message)


def _integer(value, name, minimum=1):
    _require(type(value) is int and value>=minimum, f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, minimum, maximum=None, *, inclusive=True):
    _require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value),
             f"{name} must be finite numeric")
    _require(value>=minimum if inclusive else value>minimum,f"{name} below permitted range")
    if maximum is not None:
        _require(value<=maximum if inclusive else value<maximum,f"{name} above permitted range")
    return float(value)


def validate_canonical_config(cfg):
    model=cfg["model"]; imagery=cfg["imagery"]; validation=cfg["validation"]; training=cfg["training"]
    opt=cfg["optimization"]; baseline=cfg["baseline"]; competition=cfg["competition"]
    quality=cfg["quality_training"]; decision=cfg["decision_threshold"]
    algorithms=model["algorithms"]
    _require(isinstance(algorithms,list) and len(algorithms)>0 and len(set(algorithms))==len(algorithms)
             and set(algorithms)<=set(ALGORITHMS),"algorithms must be distinct supported families")
    _integer(model["random_seed"],"random_seed",0)
    _require(model["random_seed"]<2**32,"random_seed outside supported range")
    _require(model["target_mode"]=="high_lithium_quantile","unsupported target_mode")
    _number(model["high_lithium_quantile"],"high_lithium_quantile",0,1,inclusive=False)
    fixed=model.get("fixed_threshold_ppm")
    if fixed is not None:
        _number(fixed,"fixed_threshold_ppm",0)
    _require(imagery["collection"]=="sentinel-2-l2a","only Sentinel-2 L2A is supported")
    _require(tuple(imagery["bands"])==DEFAULT_BAND_NAMES,"ordered six-band representation differs")
    pixels=_integer(imagery["patch_pixels"],"patch_pixels",2)
    size=_number(imagery["patch_size_m"],"patch_size_m",0,inclusive=False)
    _require(pixels==32 and size==640,"canonical v3/v4 patches must be 32 px / 640 m")
    _require(imagery["nodata_fill"]=="median","unsupported nodata_fill for historical v3/v4")
    _require(type(imagery["normalize_per_band"]) is bool,"normalize_per_band must be boolean")
    _number(imagery["cloud_cover_max"],"cloud_cover_max",0,100)
    _integer(imagery["search_limit"],"search_limit")
    _integer(imagery["max_retries"],"max_retries")
    _number(imagery["request_timeout_seconds"],"request_timeout_seconds",0,inclusive=False)
    _number(training["spatial_group_degrees"],"spatial_group_degrees",0,180,inclusive=False)
    _integer(training["minimum_samples"],"minimum_samples",2)
    _integer(validation["folds"],"folds",2)
    _integer(validation["inner_folds"],"inner_folds",2)
    _require(validation["prefer_spatial_groups"] is True,"canonical spatial separation is mandatory")
    _number(validation.get("minimum_distance_m",0),"minimum_distance_m",0)
    _require(baseline["enabled"] is True and baseline["strategy"]=="prior","TRAIN-only prior baseline is mandatory")
    _require(competition["primary_metric"]=="roc_auc" and competition["tie_breakers"]==["average_precision","balanced_accuracy"]
             and competition["winner_rule"]=="highest_cross_validated_roc_auc","canonical ranking is ROC-AUC, AP, BA")
    _require(type(opt["enabled"]) is bool,"optimization.enabled must be boolean")
    _integer(opt["n_trials"],"n_trials")
    for family,budget in opt.get("per_algorithm_trials",{}).items():
        _require(family in ALGORITHMS,"unknown family in trial budgets")
        _integer(budget,f"{family} budget")
    pruning=opt["pruning"]
    _require(type(pruning["enabled"]) is bool,"pruning.enabled must be boolean")
    _integer(pruning["startup_trials"],"pruning startup trials")
    _integer(pruning["warmup_folds"],"pruning warmup folds")
    _require(pruning["warmup_folds"]<=validation["inner_folds"],"pruning warmup exceeds inner folds")
    for family in algorithms:
        parameter_space_for(family,ALGORITHMS[family],opt)
    calibration_policy(opt)
    _require(decision["policy"]=="grouped_development_oof","unsupported decision policy")
    _require(decision["objective"] in {"balanced_accuracy","accuracy","f1","precision","recall","sensitivity","specificity"},"unsupported operating objective")
    # Canonical cross-fits fail on single class. The historical fallback is not used.
    _require(decision["fallback"]==.5,"fallback is fixed legacy metadata, not a canonical operating option")
    _require(type(quality["enabled"]) is bool,"quality_training.enabled must be boolean")
    _number(quality["minimum_joint_valid_fraction"],"minimum_joint_valid_fraction",0,1)
    _require(quality["minimum_joint_valid_fraction"]>0,"QA threshold must be positive")
    _require(quality["policy"]=="train_only_preserve_all_outer_test","invalid quality policy")
    v4=training.get("spectral_v4",{})
    if v4.get("enabled"):
        from lithiumscope.model_2.preparation.features.v4_inputs import ALGORITHMS as canonical
        _require(tuple(algorithms)==canonical and v4["feature_count"]==170,"v4 requires six families and 170 features")
        _require(quality["enabled"] and quality["minimum_joint_valid_fraction"]==.95,"v4 requires TRAIN-only QA95")
        _require(imagery["normalize_per_band"] is False,"v4 frozen feature matrix requires unnormalized legacy bands")
    for name in ("min_roc_auc_gain_over_baseline","min_average_precision_gain_over_baseline"):
        _number(cfg["release_gate"][name],name,0,1)
    return {"status":"validated","configuration_sha256":canonical_json_hash(cfg),
            "effective":{"algorithms":algorithms,"target_endpoint":"fixed_ppm" if fixed is not None else "fold_train_quantile",
                "fixed_threshold_ppm":fixed,"quantile":model["high_lithium_quantile"],"spatial":True,
                "baseline":"train_prior","ranking":["roc_auc_mean","average_precision_mean","balanced_accuracy_mean"],
                "nodata_fill":"median","patch_pixels":pixels,"patch_size_m":size,
                "operating_policy":decision["policy"],"operating_objective":decision["objective"],
                "fallback_used":False,"qa_train_only":quality["enabled"],"spectral_v4":bool(v4.get("enabled"))},
            "scope":"canonical development protocol; not external validation"}
