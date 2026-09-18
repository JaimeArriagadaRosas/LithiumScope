from __future__ import annotations

import pandas as pd

from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.category_cleaning")

CATEGORY_CANDIDATES = ("Geologycal_age", "Sample_type", "Rock_type", "Arc", "Domain")

AGE_GROUPS = {
    "devonian": "paleozoic",
    "carboniferous": "paleozoic",
    "permian": "paleozoic",
    "jurassic": "mesozoic",
    "cretaceous": "mesozoic",
    "paleogene": "cenozoic",
    "miocene": "cenozoic",
    "upper miocene": "cenozoic",
    "lower miocene": "cenozoic",
    "middle miocene": "cenozoic",
    "mio pliocene": "cenozoic",
    "pliocene": "cenozoic",
    "upper pliocene": "cenozoic",
    "lower piocene": "cenozoic",
    "quaternary": "cenozoic",
}

ROCK_RULES_GENERAL = (
    (("andesite", "basalt", "dacite", "rhyolite", "lava"), "volcanic"),
    (("ruff", "tuff", "ignim", "pyro"), "pyroclastic"),
    (("granite", "granodior", "gabbro", "diorite", "monzon", "tomalit", "tonalit"), "plutonic"),
    (("gneiss", "amphibolite", "granulite", "serpentinite"), "metamorphic"),
    (("sediment",), "sedimentary"),
    (("porphyr",), "porphyry"),
)

ROCK_RULES_SVM = (
    (("andesite", "basalt", "dacite", "rhyolite", "lava"), "volcanic"),
    (("ruff", "tuff", "ignim"), "pyroclastic"),
    (("granite", "granodior", "gabbro", "diorite"), "plutonic"),
    (("gneiss", "amphibolite"), "metamorphic"),
    (("sediment",), "sedimentary"),
)

SAMPLE_RULES_GENERAL = (
    (("lava",), "lava"),
    (("ignim",), "ignimbrite"),
    (("tuff", "pyro"), "pyroclastic"),
    (("dike", "intrus"), "intrusive"),
    (("brecc",), "breccia"),
    (("gneiss", "amphib", "metam"), "metamorphic"),
    (("fm", "formation", "grupo", "group"), "geologic_unit"),
)

SAMPLE_RULES_SVM = (
    (("lava",), "lava"),
    (("ignim",), "ignimbrite"),
    (("tuff",), "pyroclastic"),
    (("dike",), "intrusive"),
)


def _normalize(value) -> str:
    if pd.isna(value):
        return "unknown"
    value = str(value).strip().lower()
    return value if value else "unknown"


def _group_by_rules(value: str, rules) -> str:
    for needles, label in rules:
        if any(needle in value for needle in needles):
            return label
    return "other"


def clean_categories(frame: pd.DataFrame, model_family: str) -> pd.DataFrame:
    result = frame.copy()
    for column in CATEGORY_CANDIDATES:
        if column in result.columns:
            result[column] = result[column].map(_normalize)

    if "Geologycal_age" in result.columns:
        result["Geologycal_age"] = result["Geologycal_age"].map(
            lambda value: AGE_GROUPS.get(value, "unknown")
        )

    svm = model_family.lower() == "svm"
    if "Rock_type" in result.columns:
        rules = ROCK_RULES_SVM if svm else ROCK_RULES_GENERAL
        result["Rock_type"] = result["Rock_type"].map(lambda value: _group_by_rules(value, rules))

    if "Sample_type" in result.columns:
        rules = SAMPLE_RULES_SVM if svm else SAMPLE_RULES_GENERAL
        result["Sample_type"] = result["Sample_type"].map(
            lambda value: _group_by_rules(value, rules)
        )

    logger.info("Categorical normalization/grouping completed for model=%s", model_family)
    return result
