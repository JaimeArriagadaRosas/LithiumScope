from copy import deepcopy
import pytest
from lithiumscope.core.config import load_config
from lithiumscope.model_2.validation.configuration import validate_canonical_config


@pytest.mark.parametrize("section,key,value",[
    ("baseline","enabled",False),("baseline","strategy","uniform"),
    ("competition","primary_metric","accuracy"),("competition","tie_breakers",["f1"]),
    ("validation","prefer_spatial_groups",False),("validation","folds",1),
    ("imagery","nodata_fill","zero"),("imagery","patch_pixels",16),
    ("imagery","patch_size_m",1280),("imagery","bands",["red"]*6),
    ("imagery","normalize_per_band",True),("model","target_mode","regression"),
    ("model","high_lithium_quantile",1.),("model","fixed_threshold_ppm",-1.),
    ("model","random_seed",-1),("model","algorithms",["random_forest"]*6),
    ("optimization","n_trials",0),("quality_training","minimum_joint_valid_fraction",.9),
    ("decision_threshold","fallback",.2),("decision_threshold","objective","roc_auc"),
])
def test_invalid_or_ignored_scientific_options_are_rejected(section,key,value):
    config=deepcopy(load_config("model_2"))
    config[section][key]=value
    with pytest.raises(ValueError):
        validate_canonical_config(config)


def test_effective_contract_changes_for_supported_fixed_endpoint():
    cfg=deepcopy(load_config("model_2"))
    first=validate_canonical_config(cfg)
    cfg["model"]["fixed_threshold_ppm"]=21.6
    second=validate_canonical_config(cfg)
    assert first["configuration_sha256"]!=second["configuration_sha256"]
    assert first["effective"]["target_endpoint"]=="fold_train_quantile"
    assert second["effective"]["target_endpoint"]=="fixed_ppm"
    assert second["effective"]["fixed_threshold_ppm"]==21.6
    assert not second["effective"]["fallback_used"]
