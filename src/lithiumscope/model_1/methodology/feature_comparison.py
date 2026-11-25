"""Predeclared feature and target-transform comparisons, independent of competition defaults."""
from __future__ import annotations

from lithiumscope.experiments.shared.paired_suite import Arm
from lithiumscope.model_1.training.preprocessing.variants import ABLATIONS, feature_variant

def build_feature_arms(prepared, algorithms, *, seed=42):
    arms = []
    for algorithm in algorithms:
        reference = f"{algorithm}_reference_stable_v1"
        for variant in ABLATIONS:
            data = feature_variant(prepared, variant, preprocessing_policy="stable_v1")
            name = f"{algorithm}_{variant}_stable_v1"
            arms.append(Arm(name,algorithm,None if name==reference else reference,data.x,
                            {"variant":variant,"preprocessing":"stable_v1","seed":seed}))
        policies = ("legacy","tabnet_mask_v1") if algorithm=="tabnet" else ("legacy",)
        for policy in policies:
            data = feature_variant(prepared,"reference",preprocessing_policy=policy)
            arms.append(Arm(f"{algorithm}_reference_{policy}",algorithm,reference,data.x,
                            {"variant":"reference","preprocessing":policy,"seed":seed}))
    return arms


