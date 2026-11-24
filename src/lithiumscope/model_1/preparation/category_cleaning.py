from __future__ import annotations

import re
import pandas as pd

from lithiumscope.core.logging.session import get_logger

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
    (("whole rock", "wr"), "whole_rock"),
    (("volcanic glass", "gl"), "volcanic_glass"),
    (("lava",), "lava"),
    (("ignim",), "ignimbrite"),
    (("tuff", "pyro"), "pyroclastic"),
    (("dike", "intrus"), "intrusive"),
    (("brecc",), "breccia"),
    (("gneiss", "amphib", "metam"), "metamorphic"),
    (("fm", "formation", "grupo", "group"), "geologic_unit"),
)

SAMPLE_RULES_SVM = (
    (("whole rock", "wr"), "whole_rock"),
    (("volcanic glass", "gl"), "volcanic_glass"),
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


def _group_by_rules(value: str, rules, version: int = 2) -> str:
    if version == 1:
        for needles, label in rules:
            if any(needle in value for needle in needles):
                return label
        return "other"
    if value in {"unknown", "n/a", "na", "none", "not reported"}:
        return "unknown"
    value = value.replace("_", " ")
    for needles, label in rules:
        for needle in needles:
            if needle in {"gl", "wr", "fm"}:
                match = value == needle
            elif needle in {"ignim", "pyro", "intrus", "brecc", "amphib", "metam", "porphyr", "sediment", "granodior", "monzon", "tonalit", "tomalit"}:
                match = re.search(r"\b" + re.escape(needle) + r"\w*\b", value)
            else:
                match = re.search(r"\b" + re.escape(needle) + r"\b", value)
            if match:
                return label
    if value in {"trachyandesite", "trachybasalt", "rhyodacite", "shoshonite", "latite"} and rules == ROCK_RULES_GENERAL:
        return "volcanic"
    return "other"


def clean_categories(frame: pd.DataFrame, model_family: str, *, version: int = 2) -> pd.DataFrame:
    if version not in {1, 2}:
        raise ValueError("Unknown category rules version")
    result = frame.copy()
    for column in CATEGORY_CANDIDATES:
        if column in result.columns:
            result[column] = result[column].map(_normalize)

    if "Geologycal_age" in result.columns:
        result["Geologycal_age"] = result["Geologycal_age"].map(
            lambda value: AGE_GROUPS.get(value, "unknown")
        )

    if "Rock_type" in result.columns:
        result["Rock_type"] = result["Rock_type"].map(
            lambda value: _group_by_rules(
                value,
                ROCK_RULES_GENERAL, version,
            )
        )

    if "Sample_type" in result.columns:
        result["Sample_type"] = result["Sample_type"].map(
            lambda value: _group_by_rules(
                value,
                SAMPLE_RULES_GENERAL, version,
            )
        )

    logger.info(
        "Categorical normalization/grouping completed with common semantic rules "
        "(requested_model_family=%s)",
        model_family,
    )
    return result
