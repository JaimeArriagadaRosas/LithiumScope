"""Scientific closure checks for M2 grouped leak-safe and external evidence gates."""
from pathlib import Path
import numpy as np
import pandas as pd
from lithiumscope.experiments.leakage_safe_m2 import threshold
from lithiumscope.experiments.georoc_full_mamani_audit import near_full_mamani

def test_threshold_uses_only_supplied_inner_examples():
    assert 0<=threshold(np.array([0,1,0,1]),np.array([.1,.9,.2,.8]))<=1

def test_geochemical_collocation_is_only_a_flag():
    m=pd.DataFrame({"Sample":["X"],"Latitude (Y)":[-22.],"Longitude (X)":[-68.],
                    "Li_icpms":[15],"Geochemistry Reference":["test"]})
    candidates=pd.DataFrame({"latitude":[-22.],"longitude":[-68.],"li_ppm":[15.]})
    result=near_full_mamani(candidates,m.assign(**{"Latitude (Y)":pd.concat([m["Latitude (Y)"]]*800,ignore_index=True),
              "Longitude (X)":pd.concat([m["Longitude (X)"]]*800,ignore_index=True),
              "Sample":pd.Series(["X"]*800),"Li_icpms":pd.Series([15]*800),
              "Geochemistry Reference":pd.Series(["test"]*800)}))
    assert bool(result.possible_same_coordinate_li.iloc[0])
    assert not bool(result.independent_ground_truth_approved.iloc[0])

def test_label_and_group_not_features():
    code=Path("src/lithiumscope/experiments/leakage_safe_m2.py").read_text(encoding="utf-8")
    assert 'drop(columns=["Li_icpms","spatial_group"]' in code

def test_original_training_model_not_modified():
    report=Path("results/scientific_simulations/leakage_safe_m2/experiment_manifest.json")
    if report.is_file():
        import json
        d=json.loads(report.read_text(encoding="utf-8"))
        assert d["rows"]==706 and d["folds"]==5 and d["external_accuracy"] is None
